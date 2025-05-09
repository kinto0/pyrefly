/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import { ExtensionContext, workspace } from 'vscode';
import * as vscode from 'vscode';
import {
    CancellationToken,
    ConfigurationItem,
    ConfigurationParams,
    ConfigurationRequest,
    DidChangeConfigurationNotification,
    LanguageClient,
    LanguageClientOptions,
    LSPAny,
    ResponseError,
    ServerOptions,
} from 'vscode-languageclient/node';
import {PythonExtension} from '@vscode/python-extension';

let client: LanguageClient;

/// Get a setting at the path, or throw an error if it's not set.
function requireSetting<T>(path: string): T {
    const ret: T = vscode.workspace.getConfiguration().get(path);
    if (ret == undefined) {
        throw new Error(`Setting "${path}" was not configured`)
    }
    return ret;
}

 /**
   * This function adds the pythonPath to any section with configuration of 'python'.
   * Our language server expects the pythonPath from VSCode configurations but this setting is not stored in VSCode
   * configurations. The Python extension used to store pythonPath in this section but no longer does. Details:
   * https://github.com/microsoft/pyright/commit/863721687bc85a54880423791c79969778b19a3f
   *
   * Example:
   * - Pyrefly asks for a configurationItem for {scopeUri: '/home/project', section: 'python'}
   * - VSCode returns a configuration of {setting: 'value'} from settings.json
   * - This function will add pythonPath: '/usr/bin/python3' from the Python extension to the configuration
   * - {setting: 'value', pythonPath: '/usr/bin/python3'} is returned
   *
   * @param pythonExtension the python extension API
   * @param configurationItems the sections within the workspace
   * @param configuration the configuration returned by vscode in response to a workspace/configuration request (usually what's in settings.json)
   * corresponding to the sections described in configurationItems
   */
 async function overridePythonPath(
    pythonExtension: PythonExtension,
    configurationItems: ConfigurationItem[],
    configuration: (object | null)[],
  ): Promise<(object | null)[]> {
    const getPythonPathForConfigurationItem = async (index: number) => {
      if (configurationItems.length <= index || configurationItems[index].section !== 'python') {
        return undefined;
      }
      let scopeUri = configurationItems[index].scopeUri;
      return await pythonExtension.environments.getActiveEnvironmentPath(scopeUri === undefined ? undefined : vscode.Uri.file(scopeUri)).path;
    };
    const newResult = await Promise.all(configuration.map(async (item, index) => {
      const pythonPath = await getPythonPathForConfigurationItem(index);
      if (pythonPath === undefined) {
        return item;
      } else {
        return {...item, pythonPath};
      }
    }));
    return newResult;
  }

export async function activate(context: ExtensionContext) {
    const path: string = requireSetting("pyrefly.lspPath");
    const args: [string] = requireSetting("pyrefly.lspArguments");

    const bundledPyreflyPath = vscode.Uri.joinPath(
        context.extensionUri,
        'bin',
        'release',
        // process.platform returns win32 on any windows CPU architecture
        process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly'
    );

    let pythonExtension = await PythonExtension.api();

    // Otherwise to spawn the server
    let serverOptions: ServerOptions = { command: path === '' ? bundledPyreflyPath.fsPath : path, args: args };
    let rawInitialisationOptions = {
      ...vscode.workspace.getConfiguration("pyrefly"),
      // allows us to open a text document that does not exist on disk using the `contentsasuri` scheme from Pyrefly
      supportContentsAsUri: true
    };

    // Options to control the language client
    let clientOptions: LanguageClientOptions = {
        initializationOptions: rawInitialisationOptions,
        // Register the server for Starlark documents
        documentSelector: [{ scheme: 'file', language: 'python' }],
        middleware: {
            workspace: {
                configuration: async (
                    params: ConfigurationParams,
                    token: CancellationToken,
                    next: ConfigurationRequest.HandlerSignature,
                  ): Promise<LSPAny[] | ResponseError<void>> => {
                    const result = await next(params, token);
                    if (result instanceof ResponseError) {
                      return result;
                    }
                    const newResult = await overridePythonPath(pythonExtension, params.items, result as (object | null)[]);
                    return newResult;
                  },
            }
        }
    };

    // Create the language client and start the client.
    client = new LanguageClient(
        'pyrefly',
        'Pyrefly language server',
        serverOptions,
        clientOptions
    );

    context.subscriptions.push(
        pythonExtension.environments.onDidChangeActiveEnvironmentPath(() => {
          client.sendNotification(DidChangeConfigurationNotification.type, {settings: {}});
        })
    );

    context.subscriptions.push(
      workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration("python.pyrefly")) {
          client.sendNotification(DidChangeConfigurationNotification.type, {settings: {}});
        }
      }));

    context.subscriptions.push(
        vscode.commands.registerCommand('pyrefly.restartClient', async () => {
            await client.stop();
            client = new LanguageClient(
                'pyrefly',
                'Pyrefly language server',
                serverOptions,
                clientOptions
            );
            await client.start();
        }),
    );

    registerTextDocumentContentProviders();

    // When our extension is activated, make sure ms-python knows
    // TODO(kylei): remove this hack once ms-python has this behavior
    await triggerMsPythonRefreshLanguageServers();

    vscode.workspace.onDidChangeConfiguration(async (e) => {
      if (e.affectsConfiguration(`python.pyrefly.disableLanguageServices`)) {
        // TODO(kylei): remove this hack once ms-python has this behavior
        await triggerMsPythonRefreshLanguageServers();
      }
    })

    // Start the client. This will also launch the server
    await client.start();
}

/**
 * VSCode allows registering a content provider for a URI scheme. This allows us to
 * open a text document that does not necessarily exist on disk using a specific schema.
 */
function registerTextDocumentContentProviders() {
  // This document provider encodes the entire text document contents as the query section of the URI.
  const provider = new (class implements vscode.TextDocumentContentProvider {
    provideTextDocumentContent(uri: vscode.Uri): string {
      return Buffer.from(uri.query, 'base64').toString();
    }
  })();

  vscode.workspace.registerTextDocumentContentProvider("contentsasuri", provider);
}

/**
 * This function will trigger the ms-python extension to reasses which language server to spin up.
 * It does this by changing languageServer setting: this triggers a refresh of active language
 * servers:
 * https://github.com/microsoft/vscode-python/blob/main/src/client/languageServer/watcher.ts#L296
 *
 * We then change the setting back so we don't end up messing up the users settings.
 */
async function triggerMsPythonRefreshLanguageServers() {
    const config = vscode.workspace.getConfiguration('python');
    const setting = 'languageServer';
    let previousSetting = config.get(setting);
    // without the target, we will crash here with "Unable to write to Workspace Settings
    // because no workspace is opened. Please open a workspace first and try again."
    await config.update(setting, previousSetting === 'None' ? 'Default' : 'None', vscode.ConfigurationTarget.Global);
    await config.update(setting, previousSetting, vscode.ConfigurationTarget.Global);
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
