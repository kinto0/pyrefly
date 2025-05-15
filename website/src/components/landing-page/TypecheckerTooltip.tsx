/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import {
    Project,
    TypeChecker,
    type TypeCheckerValue,
} from './PerformanceComparisonTypes';
import Tooltip from './Tooltip';

interface TypecheckerTooltipProps {
    typechecker: TypeCheckerValue;
    project: Project;
}

export default function TypecheckerTooltip({
    typechecker,
    project,
}: TypecheckerTooltipProps): React.ReactElement {
    return <Tooltip content={getTypecheckerInfo(typechecker, project)} />;
}

export function getTypecheckerInfo(
    typechecker: TypeCheckerValue,
    project: Project
): React.ReactNode {
    switch (typechecker) {
        case TypeChecker.PYREFLY:
            // Include branching for Pyrefly since we only run Pyrefly on more than one project
            switch (project) {
                case Project.INSTAGRAM:
                    return (
                        <>
                            <strong>Command: "pyrefly check"</strong>
                            <br />
                            Pyrefly uses as many threads as possible
                        </>
                    );
                case Project.PYTORCH:
                    return (
                        <>
                            <strong>Command: "pyrefly check"</strong>
                            <br />
                            Pyrefly uses as many threads as possible
                        </>
                    );
            }
        case TypeChecker.PYRE:
            return (
                <>
                    <strong>Command: "pyre --no-saved-state check"</strong>
                    <br />
                    Pyre uses as many threads as possible
                </>
            );
        case TypeChecker.PYRIGHT:
            return (
                <>
                    <strong>Command: "pyright --threads=8"</strong>
                    <br />8 threads yielded the best performance after testing
                    multiple settings
                </>
            );
        case TypeChecker.MYPY:
            return (
                <>
                    <strong>Command: "dmypy run"</strong>
                </>
            );
    }
}
