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
import type { ProjectValue } from './PerformanceComparisonTypes';
import typography from './typography';
import { landingPageCardStyles } from './landingPageCardStyles';
interface PerformanceComparisonButtonProps {
    project: ProjectValue;
    selectedProject: ProjectValue;
    setSelectedProject: React.Dispatch<React.SetStateAction<ProjectValue>>;
}

export default function PerformanceComparisonButton({
    project,
    selectedProject,
    setSelectedProject,
}: PerformanceComparisonButtonProps): React.ReactElement {
    return (
        <button
            {...stylex.props(
                landingPageCardStyles.card,
                styles.button,
                typography.p,
                selectedProject === project && styles.buttonActive
            )}
            onClick={() => setSelectedProject(project)}
        >
            {project}
        </button>
    );
}

const styles = stylex.create({
    button: {
        padding: '0.75rem 1.5rem',
        backgroundColor: 'var(--color-background)',
        cursor: 'pointer',
        ':hover': {
            transform: 'translateY(-5px)',
        },
    },
    buttonActive: {
        backgroundColor: 'var(--color-background)',
        color: 'var(--color-primary)',
    },
});
