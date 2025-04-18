/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import { ExtensionContext } from 'vscode';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

/// Get a setting at the path, or throw an error if it's not set.
function requireSetting<T>(path: string): T {
    const ret: T = vscode.workspace.getConfiguration().get(path);
    if (ret == undefined) {
        throw new Error(`Setting "${path}" was not configured`)
    }
    return ret;
}

export function activate(context: ExtensionContext) {
    const path: string = requireSetting("pyrefly.lspPath");
    const args: [string] = requireSetting("pyrefly.lspArguments");

    const bundledPyreflyPath = vscode.Uri.joinPath(
        context.extensionUri,
        'bin',
        'release',
        // process.platform returns win32 on any windows CPU architecture
        process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly'
    );

    // Otherwise to spawn the server
    let serverOptions: ServerOptions = { command: path === '' ? bundledPyreflyPath.fsPath : path, args: args };
    let rawInitialisationOptions = vscode.workspace.getConfiguration("pyrefly");

    // Options to control the language client
    let clientOptions: LanguageClientOptions = {
        initializationOptions: rawInitialisationOptions,
        // Register the server for Starlark documents
        documentSelector: [{ scheme: 'file', language: 'python' }],
    };

    // Create the language client and start the client.
    client = new LanguageClient(
        'pyrefly',
        'Pyrefly language server',
        serverOptions,
        clientOptions
    );

    // Start the client. This will also launch the server
    client.start();

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
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
