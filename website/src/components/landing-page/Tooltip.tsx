/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import * as stylex from '@stylexjs/stylex';
import typography from './typography';

interface TooltipProps {
    content: React.ReactNode;
}

export default function Tooltip({ content }: TooltipProps): React.ReactElement {
    return (
        <span {...stylex.props(styles.tooltipContainer)}>
            <span
                {...stylex.props(typography.p, styles.infoIcon)}
                onMouseEnter={(e) => {
                    const tooltip = e.currentTarget.nextElementSibling;
                    if (tooltip) {
                        (tooltip as HTMLElement).style.visibility = 'visible';
                        (tooltip as HTMLElement).style.opacity = '1';
                    }
                }}
                onMouseLeave={(e) => {
                    const tooltip = e.currentTarget.nextElementSibling;
                    if (tooltip) {
                        (tooltip as HTMLElement).style.visibility = 'hidden';
                        (tooltip as HTMLElement).style.opacity = '0';
                    }
                }}
            >
                ⓘ
            </span>
            <span {...stylex.props(styles.tooltip, typography.p)}>
                {content}
            </span>
        </span>
    );
}

const styles = stylex.create({
    tooltipContainer: {
        position: 'relative',
        display: 'inline-block',
        marginLeft: '8px',
    },
    infoIcon: {
        fontSize: '12px',
        color: 'var(--color-text)',
        cursor: 'help',
        verticalAlign: 'middle',
    },
    tooltip: {
        visibility: 'hidden',
        backgroundColor: 'var(--color-background-secondary)',
        color: 'var(--color-text)',
        textAlign: 'center',
        borderRadius: '6px',
        padding: '8px 12px',
        border: '1px solid var(--color-background)',
        position: 'absolute',
        zIndex: 1,
        bottom: '125%',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '300px',
        opacity: 0,
        transition: 'opacity 0.3s',
        whiteSpace: 'normal',
        wordWrap: 'break-word',
    },
});
