/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import { Project, type ProjectValue } from './PerformanceComparisonTypes';
import Tooltip from './Tooltip';

interface PerformanceDescriptionTooltipProps {
    project: ProjectValue;
}

export default function PerformanceDescriptionTooltip({
    project,
}: PerformanceDescriptionTooltipProps): React.ReactElement {
    return <Tooltip content={getInfrastructureInfo(project)} />;
}

function getInfrastructureInfo(project: ProjectValue): React.ReactNode {
    switch (project) {
        case Project.INSTAGRAM:
            return (
                <>
                    <strong>Tested using Meta infrastructure</strong>
                    <br />
                    (166 cores, 228 GB RAM)
                </>
            );
        case Project.PYTORCH:
            return (
                <>
                    <strong>Tested using Macbook</strong>
                    <br />
                    (10 cores: 8 performance + 2 efficiency cores, 32 GB RAM)
                </>
            );
    }
}
