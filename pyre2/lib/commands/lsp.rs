/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use dupe::Dupe;
use lsp_server::Connection;
use lsp_server::ErrorCode;
use lsp_server::Message;
use lsp_server::Notification;
use lsp_server::Request;
use lsp_server::RequestId;
use lsp_server::Response;
use lsp_server::ResponseError;
use lsp_types::notification::Cancel;
use lsp_types::notification::DidChangeTextDocument;
use lsp_types::notification::DidCloseTextDocument;
use lsp_types::notification::DidOpenTextDocument;
use lsp_types::notification::PublishDiagnostics;
use lsp_types::request::Completion;
use lsp_types::request::GotoDefinition;
use lsp_types::request::HoverRequest;
use lsp_types::request::InlayHintRequest;
use lsp_types::CompletionList;
use lsp_types::CompletionOptions;
use lsp_types::CompletionParams;
use lsp_types::CompletionResponse;
use lsp_types::Diagnostic;
use lsp_types::DidChangeTextDocumentParams;
use lsp_types::DidCloseTextDocumentParams;
use lsp_types::DidOpenTextDocumentParams;
use lsp_types::GotoDefinitionParams;
use lsp_types::GotoDefinitionResponse;
use lsp_types::Hover;
use lsp_types::HoverContents;
use lsp_types::HoverParams;
use lsp_types::HoverProviderCapability;
use lsp_types::InitializeParams;
use lsp_types::InlayHint;
use lsp_types::InlayHintLabel;
use lsp_types::InlayHintParams;
use lsp_types::Location;
use lsp_types::MarkupContent;
use lsp_types::MarkupKind;
use lsp_types::NumberOrString;
use lsp_types::OneOf;
use lsp_types::PublishDiagnosticsParams;
use lsp_types::Range;
use lsp_types::ServerCapabilities;
use lsp_types::TextDocumentSyncCapability;
use lsp_types::TextDocumentSyncKind;
use lsp_types::TextEdit;
use lsp_types::Url;
use ruff_source_file::SourceLocation;
use ruff_text_size::TextSize;
use serde::de::DeserializeOwned;
use starlark_map::small_map::SmallMap;

use crate::clap_env;
use crate::commands::util::module_from_path;
use crate::config::ErrorConfigs;
use crate::metadata::RuntimeMetadata;
use crate::module::bundled::typeshed;
use crate::module::finder::find_module;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::SourceRange;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::run::CommandExitStatus;
use crate::state::handle::Handle;
use crate::state::loader::FindError;
use crate::state::loader::Loader;
use crate::state::loader::LoaderId;
use crate::state::require::Require;
use crate::state::state::State;
use crate::util::lock::Mutex;
use crate::util::prelude::VecExt;

#[derive(Debug, Parser, Clone)]
pub struct Args {
    #[clap(long = "search-path", env = clap_env("SEARCH_PATH"))]
    pub(crate) search_path: Vec<PathBuf>,
    #[clap(long = "site-package-path", env = clap_env("SITE_PACKAGE_PATH"))]
    pub(crate) site_package_path: Vec<PathBuf>,
}

struct Server<'a> {
    send: &'a dyn Fn(Message),
    #[expect(dead_code)] // we'll use it later on
    initialize_params: InitializeParams,
    search_path: Vec<PathBuf>,
    state: Mutex<State>,
    config: RuntimeMetadata,
    loader: LoaderId,
    canceled_requests: HashSet<RequestId>,
    open_files: Arc<Mutex<SmallMap<PathBuf, (i32, Arc<String>)>>>,
}

pub fn run_lsp(
    connection: &Connection,
    wait_on_connection: impl FnOnce() -> anyhow::Result<()>,
    args: Args,
) -> anyhow::Result<CommandExitStatus> {
    // Run the server and wait for the two threads to end (typically by trigger LSP Exit event).
    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_owned()]),
            ..Default::default()
        }),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        ..Default::default()
    })
    .unwrap();
    let initialization_params = match connection.initialize(server_capabilities) {
        Ok(it) => serde_json::from_value(it).unwrap(),
        Err(e) => {
            // Use this in later versions of LSP server
            // if e.channel_is_disconnected() {
            // io_threads.join()?;
            // }
            return Err(e.into());
        }
    };
    let search_path = args.search_path;
    let site_package_path = args.site_package_path;
    let send = |msg| connection.sender.send(msg).unwrap();
    let mut server = Server::new(&send, initialization_params, search_path, site_package_path);
    eprintln!("Reading messages");
    for msg in &connection.receiver {
        if matches!(&msg, Message::Request(req) if connection.handle_shutdown(req)?) {
            break;
        }
        server.process(msg)?;
    }
    wait_on_connection()?;

    // Shut down gracefully.
    eprintln!("shutting down server");
    Ok(CommandExitStatus::Success)
}

impl Args {
    pub fn run(mut self, extra_search_paths: Vec<PathBuf>) -> anyhow::Result<CommandExitStatus> {
        // Note that  we must have our logging only write out to stderr.
        eprintln!("starting generic LSP server");

        // Create the transport. Includes the stdio (stdin and stdout) versions but this could
        // also be implemented to use sockets or HTTP.
        let (connection, io_threads) = Connection::stdio();

        self.search_path.extend(extra_search_paths);
        run_lsp(
            &connection,
            move || io_threads.join().map_err(anyhow::Error::from),
            self,
        )
    }
}

#[derive(Debug, Clone)]
struct LspLoader {
    open_files: Arc<Mutex<SmallMap<PathBuf, (i32, Arc<String>)>>>,
    search_path: Vec<PathBuf>,
    site_package_path: Vec<PathBuf>,
}

impl Loader for LspLoader {
    fn find_import(&self, module: ModuleName) -> Result<ModulePath, FindError> {
        if let Some(path) = find_module(module, &self.search_path) {
            Ok(path)
        } else if let Some(path) = typeshed().map_err(FindError::new)?.find(module) {
            Ok(path)
        } else if let Some(path) = find_module(module, &self.site_package_path) {
            Ok(path)
        } else {
            Err(FindError::search_path(
                &self.search_path,
                &self.site_package_path,
            ))
        }
    }

    fn load_from_memory(&self, path: &Path) -> Option<Arc<String>> {
        Some(self.open_files.lock().get(path)?.1.dupe())
    }
}

/// Convert to a path we can show to the user. The contents may not match the disk, but it has
/// to be basically right.
fn to_real_path(path: &ModulePath) -> Option<&Path> {
    match path.details() {
        ModulePathDetails::FileSystem(path)
        | ModulePathDetails::Memory(path)
        | ModulePathDetails::Namespace(path) => Some(path),
        ModulePathDetails::BundledTypeshed(_) => None,
    }
}

impl<'a> Server<'a> {
    fn process(&mut self, msg: Message) -> anyhow::Result<()> {
        match msg {
            Message::Request(x) => {
                if self.canceled_requests.remove(&x.id) {
                    let message = format!("Request {} is canceled", x.id);
                    eprintln!("{message}");
                    self.send_response(Response::new_err(
                        x.id,
                        ErrorCode::RequestCanceled as i32,
                        message,
                    ));
                    return Ok(());
                }
                eprintln!("Handling non-canceled request ({})", x.id);
                if let Some(params) = as_request::<GotoDefinition>(&x) {
                    let default_response = GotoDefinitionResponse::Array(Vec::new());
                    self.send_response(new_response(
                        x.id,
                        Ok(self.goto_definition(params).unwrap_or(default_response)),
                    ));
                } else if let Some(params) = as_request::<Completion>(&x) {
                    self.send_response(new_response(x.id, self.completion(params)));
                } else if let Some(params) = as_request::<HoverRequest>(&x) {
                    let default_response = Hover {
                        contents: HoverContents::Array(Vec::new()),
                        range: None,
                    };
                    self.send_response(new_response(
                        x.id,
                        Ok(self.hover(params).unwrap_or(default_response)),
                    ));
                } else if let Some(params) = as_request::<InlayHintRequest>(&x) {
                    self.send_response(new_response(
                        x.id,
                        Ok(self.inlay_hints(params).unwrap_or_default()),
                    ))
                } else {
                    eprintln!("Unhandled request: {x:?}");
                }
                Ok(())
            }
            Message::Response(x) => {
                eprintln!("Unhandled response: {x:?}");
                Ok(())
            }
            Message::Notification(x) => {
                if let Some(params) = as_notification::<DidOpenTextDocument>(&x) {
                    self.did_open(params)
                } else if let Some(params) = as_notification::<DidChangeTextDocument>(&x) {
                    self.did_change(params)
                } else if let Some(params) = as_notification::<DidCloseTextDocument>(&x) {
                    self.did_close(params)
                } else if let Some(params) = as_notification::<Cancel>(&x) {
                    let id = match params.id {
                        NumberOrString::Number(i) => RequestId::from(i),
                        NumberOrString::String(s) => RequestId::from(s),
                    };
                    self.canceled_requests.insert(id);
                    Ok(())
                } else {
                    eprintln!("Unhandled notification: {x:?}");
                    Ok(())
                }
            }
        }
    }

    fn new(
        send: &'a dyn Fn(Message),
        initialize_params: InitializeParams,
        search_path: Vec<PathBuf>,
        site_package_path: Vec<PathBuf>,
    ) -> Self {
        let open_files = Arc::new(Mutex::new(SmallMap::new()));
        let loader = LoaderId::new(LspLoader {
            open_files: open_files.dupe(),
            search_path: search_path.clone(),
            site_package_path,
        });
        Self {
            send,
            initialize_params,
            search_path,
            state: Mutex::new(State::new()),
            config: RuntimeMetadata::default(),
            loader,
            canceled_requests: HashSet::new(),
            open_files,
        }
    }

    fn send_notification(&self, x: Notification) {
        (self.send)(Message::Notification(x))
    }

    fn send_response(&self, x: Response) {
        (self.send)(Message::Response(x))
    }

    fn publish_diagnostics(&self, uri: Url, diags: Vec<Diagnostic>, version: Option<i32>) {
        self.send_notification(new_notification::<PublishDiagnostics>(
            PublishDiagnosticsParams::new(uri, diags, version),
        ));
    }

    fn validate(&self) -> anyhow::Result<()> {
        let handles = self
            .open_files
            .lock()
            .keys()
            .map(|x| {
                (
                    Handle::new(
                        module_from_path(x, &self.search_path),
                        ModulePath::memory(x.clone()),
                        self.config.dupe(),
                        self.loader.dupe(),
                    ),
                    Require::Everything,
                )
            })
            .collect::<Vec<_>>();

        let state = self.state.lock();
        let mut transaction = state.new_committable_transaction(Require::Exports, None);
        transaction.as_mut().invalidate_memory(
            self.loader.dupe(),
            &self.open_files.lock().keys().cloned().collect::<Vec<_>>(),
        );
        state.run_with_committing_transaction(transaction, &handles);
        let mut diags: SmallMap<PathBuf, Vec<Diagnostic>> = SmallMap::new();
        let open_files = self.open_files.lock();
        for x in open_files.keys() {
            diags.insert(x.as_path().to_owned(), Vec::new());
        }
        // TODO(connernilsen): replace with real error config from config file
        for e in state
            .transaction()
            .readable()
            .get_loads(handles.iter().map(|(handle, _)| handle))
            .collect_errors(&ErrorConfigs::default())
            .shown
        {
            if let Some(path) = to_real_path(e.path()) {
                if open_files.contains_key(path) {
                    diags.entry(path.to_owned()).or_default().push(Diagnostic {
                        range: source_range_to_range(e.source_range()),
                        severity: Some(lsp_types::DiagnosticSeverity::ERROR),
                        source: Some("Pyre2".to_owned()),
                        message: e.msg().to_owned(),
                        code: Some(lsp_types::NumberOrString::String(
                            e.error_kind().to_name().to_owned(),
                        )),
                        ..Default::default()
                    });
                }
            }
        }
        for (path, diags) in diags {
            let path = std::fs::canonicalize(&path).unwrap_or(path);
            match Url::from_file_path(&path) {
                Ok(uri) => self.publish_diagnostics(uri, diags, None),
                Err(_) => eprint!("Unable to convert path to uri: {path:?}"),
            }
        }
        Ok(())
    }

    fn did_open(&self, params: DidOpenTextDocumentParams) -> anyhow::Result<()> {
        self.open_files.lock().insert(
            params.text_document.uri.to_file_path().unwrap(),
            (
                params.text_document.version,
                Arc::new(params.text_document.text),
            ),
        );
        self.validate()
    }

    fn did_change(&self, params: DidChangeTextDocumentParams) -> anyhow::Result<()> {
        // We asked for Sync full, so can just grab all the text from params
        let change = params.content_changes.into_iter().next().unwrap();
        self.open_files.lock().insert(
            params.text_document.uri.to_file_path().unwrap(),
            (params.text_document.version, Arc::new(change.text)),
        );
        self.validate()
    }

    fn did_close(&self, params: DidCloseTextDocumentParams) -> anyhow::Result<()> {
        self.open_files
            .lock()
            .shift_remove(&params.text_document.uri.to_file_path().unwrap());
        self.publish_diagnostics(params.text_document.uri, Vec::new(), None);
        Ok(())
    }

    fn make_handle(&self, uri: &Url) -> Handle {
        let path = uri.to_file_path().unwrap();
        let module = module_from_path(&path, &self.search_path);
        let module_path = if self.open_files.lock().contains_key(&path) {
            ModulePath::memory(path)
        } else {
            ModulePath::filesystem(path)
        };
        Handle::new(module, module_path, self.config.dupe(), self.loader.dupe())
    }

    fn goto_definition(&self, params: GotoDefinitionParams) -> Option<GotoDefinitionResponse> {
        let state = self.state.lock();
        let transaction = state.transaction();
        let handle = self.make_handle(&params.text_document_position_params.text_document.uri);
        let info = transaction.readable().get_module_info(&handle)?;
        let range = position_to_text_size(&info, params.text_document_position_params.position);
        let TextRangeWithModuleInfo {
            module_info: definition_module_info,
            range,
        } = transaction.goto_definition(&handle, range)?;
        let path = to_real_path(definition_module_info.path())?;
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned());
        Some(GotoDefinitionResponse::Scalar(Location {
            uri: Url::from_file_path(path).unwrap(),
            range: source_range_to_range(&definition_module_info.source_range(range)),
        }))
    }

    fn completion(&self, params: CompletionParams) -> anyhow::Result<CompletionResponse> {
        let state = self.state.lock();
        let transaction = state.transaction();
        let handle = self.make_handle(&params.text_document_position.text_document.uri);
        let items = transaction
            .readable()
            .get_module_info(&handle)
            .map(|info| {
                transaction.completion(
                    &handle,
                    position_to_text_size(&info, params.text_document_position.position),
                )
            })
            .unwrap_or_default();
        Ok(CompletionResponse::List(CompletionList {
            is_incomplete: false,
            items,
        }))
    }

    fn hover(&self, params: HoverParams) -> Option<Hover> {
        let state = self.state.lock();
        let handle = self.make_handle(&params.text_document_position_params.text_document.uri);
        let transaction = state.transaction();
        let info = transaction.readable().get_module_info(&handle)?;
        let range = position_to_text_size(&info, params.text_document_position_params.position);
        let t = transaction.hover(&handle, range)?;
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    r#"```python
{}
```"#,
                    t
                ),
            }),
            range: None,
        })
    }

    fn inlay_hints(&self, params: InlayHintParams) -> Option<Vec<InlayHint>> {
        let state = self.state.lock();
        let handle = self.make_handle(&params.text_document.uri);
        let transaction = state.transaction();
        let info = transaction.readable().get_module_info(&handle)?;
        let t = transaction.inlay_hints(&handle)?;
        Some(t.into_map(|x| {
            let position = text_size_to_position(&info, x.0);
            InlayHint {
                position,
                label: InlayHintLabel::String(x.1.clone()),
                kind: None,
                text_edits: Some(vec![TextEdit {
                    range: Range::new(position, position),
                    new_text: x.1,
                }]),
                tooltip: None,
                padding_left: None,
                padding_right: None,
                data: None,
            }
        }))
    }
}

fn source_range_to_range(x: &SourceRange) -> lsp_types::Range {
    lsp_types::Range::new(
        source_location_to_position(&x.start),
        source_location_to_position(&x.end),
    )
}

fn source_location_to_position(x: &SourceLocation) -> lsp_types::Position {
    lsp_types::Position {
        line: x.row.to_zero_indexed() as u32,
        character: x.column.to_zero_indexed() as u32,
    }
}

fn text_size_to_position(info: &ModuleInfo, x: TextSize) -> lsp_types::Position {
    source_location_to_position(&info.source_location(x))
}

fn position_to_text_size(info: &ModuleInfo, position: lsp_types::Position) -> TextSize {
    info.to_text_size(position.line, position.character)
}

fn as_notification<T>(x: &Notification) -> Option<T::Params>
where
    T: lsp_types::notification::Notification,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD {
        let params = serde_json::from_value(x.params.clone()).unwrap_or_else(|err| {
            panic!(
                "Invalid notification\nMethod: {}\n error: {}",
                x.method, err
            )
        });
        Some(params)
    } else {
        None
    }
}

fn as_request<T>(x: &Request) -> Option<T::Params>
where
    T: lsp_types::request::Request,
    T::Params: DeserializeOwned,
{
    if x.method == T::METHOD {
        let params = serde_json::from_value(x.params.clone()).unwrap_or_else(|err| {
            panic!(
                "Invalid request\n  method: {}\n  error: {}\n  request: {:?}\n",
                x.method, err, x
            )
        });
        Some(params)
    } else {
        None
    }
}

/// Create a new `Notification` object with the correct name from the given params.
fn new_notification<T>(params: T::Params) -> Notification
where
    T: lsp_types::notification::Notification,
{
    Notification {
        method: T::METHOD.to_owned(),
        params: serde_json::to_value(&params).unwrap(),
    }
}

fn new_response<T>(id: RequestId, params: anyhow::Result<T>) -> Response
where
    T: serde::Serialize,
{
    match params {
        Ok(params) => Response {
            id,
            result: Some(serde_json::to_value(params).unwrap()),
            error: None,
        },
        Err(e) => Response {
            id,
            result: None,
            error: Some(ResponseError {
                code: 0,
                message: format!("{:#?}", e),
                data: None,
            }),
        },
    }
}
