/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';

// Mock implementation of ThemedImage component
const ThemedImage: React.FC<{
    sources: {
        light: string;
        dark: string;
    };
    alt?: string;
    className?: string;
    [key: string]: any;
}> = (props) => {
    // Use the light source by default in tests
    const { sources, alt, className, ...restProps } = props;

    return (
        <img
            src={sources.light}
            alt={alt || ''}
            className={className}
            {...restProps}
        />
    );
};

export default ThemedImage;
