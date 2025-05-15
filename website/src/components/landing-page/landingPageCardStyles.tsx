/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as stylex from '@stylexjs/stylex';

// Reusable card styles that can be used across different components
export const landingPageCardStyles = stylex.create({
    card: {
        padding: '1.75rem',
        background: 'var(--color-landing-page-card-background)',
        backdropFilter: 'blur(10px)',
        borderRadius: '8px',
        border: '1px solid var(--color-landing-page-card-border)',
        boxShadow:
            '0 4px 6px var(--color-shadow), 0 1px 3px var(--color-shadow-background)',
        transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        ':hover': {
            boxShadow:
                '0 10px 20px var(--color-shadow-hovered), 0 3px 6px var(--color-shadow)',
            background: 'var(--color-landing-page-card-background-hovered)',
        },
    },
});
