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
import { Project, ProjectValue } from './PerformanceComparisonTypes';
import PerformanceDescriptionTooltip from './PerformanceDescriptionTooltip';

interface PerformanceComparisonChartProps {
    project: ProjectValue;
}

export default function PerformanceComparisonDescription({
    project,
}: PerformanceComparisonChartProps): React.ReactElement {
    return (
        <div {...stylex.props(styles.messageContainer, typography.h5)}>
            {getDescriptionText(project)}
        </div>
    );
}

function getDescriptionText(project: ProjectValue): React.ReactNode {
    switch (project) {
        case Project.INSTAGRAM:
            return (
                <>
                    Type checking the Instagram codebase from scratch.
                    <PerformanceDescriptionTooltip project={project} />
                </>
            );
        case Project.PYTORCH:
            return (
                <>
                    Type checking the{' '}
                    <a
                        href="https://github.com/pytorch/pytorch"
                        target="_blank"
                        rel="noopener noreferrer"
                        {...stylex.props(styles.link)}
                    >
                        PyTorch
                    </a>{' '}
                    codebase from scratch.
                    <PerformanceDescriptionTooltip project={project} />
                </>
            );
    }
}

const styles = stylex.create({
    messageContainer: {
        paddingBottom: '2rem',
    },
    link: {
        color: 'var(--color-primary)',
        ':hover': {
            color: 'var(--color-primary-hover)',
        },
    },
});
