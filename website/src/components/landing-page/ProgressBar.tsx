/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import * as stylex from '@stylexjs/stylex';

interface ProgressBarProps {
    durationInSeconds: number;
    maxDurationInSeconds: number;
    highlight: boolean;
    isLoaded: boolean;
}

export default function ProgressBar({
    durationInSeconds,
    maxDurationInSeconds,
    highlight,
    isLoaded,
}: ProgressBarProps): React.ReactElement {
    // Calculate the relative percentage width based on the maximum duration
    const relativeWidth = (durationInSeconds / maxDurationInSeconds) * 100;

    return (
        <div
            {...stylex.props(styles.progressBar)}
            style={{
                height: '80%',
                width: `${relativeWidth}%`,
            }}
        >
            <div
                {...stylex.props(
                    styles.fill,
                    highlight && styles.highlight,
                    isLoaded && styles.animate
                )}
                style={{
                    animationDuration: `${durationInSeconds}s`,
                    width: isLoaded ? undefined : '0%',
                }}
            />
        </div>
    );
}

const fillUpKeyframes = stylex.keyframes({
    '0%': {
        width: '0%',
    },
    '100%': {
        width: '100%',
    },
});

const styles = stylex.create({
    progressBar: {
        backgroundColor: 'var(--light-grey)',
        borderRadius: 30,
        width: '100%',
        overflow: 'hidden',
    },
    fill: {
        height: '100%',
        backgroundColor: 'var(--color-secondary)',
        borderRadius: 'inherit',
        animationFillMode: 'forwards',
        animationTimingFunction: 'linear',
    },
    animate: {
        animationName: fillUpKeyframes,
    },
    highlight: {
        backgroundColor: 'var(--color-primary)',
    },
});
