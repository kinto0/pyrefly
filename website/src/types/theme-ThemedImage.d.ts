/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

declare module '@theme/ThemedImage' {
    import type { ComponentProps, ReactNode } from 'react';

    export interface Props extends Omit<ComponentProps<'img'>, 'src'> {
        readonly sources: {
            readonly light: string;
            readonly dark: string;
        };
    }

    export default function ThemedImage(props: Props): ReactNode;
}
