/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

(function () {
    // Function to update favicon based on color mode
    function updateFavicon(isDarkMode) {
        const faviconLink = document.querySelector('link[rel="icon"]');
        if (faviconLink) {
            // Get the baseUrl from Docusaurus
            const baseUrl =
                document
                    .querySelector('meta[name="docusaurus-base-url"]')
                    ?.getAttribute('content') || '/';

            faviconLink.href = isDarkMode
                ? baseUrl + 'img/Pyrefly-Symbol-Invert.svg'
                : baseUrl + 'img/Pyrefly-Symbol.svg';
        }
    }

    // Initial update based on current mode
    const isDarkMode = document.documentElement.dataset.theme === 'dark';
    updateFavicon(isDarkMode);

    // Set up a MutationObserver to watch for theme changes
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'data-theme') {
                const isDarkMode =
                    document.documentElement.dataset.theme === 'dark';
                updateFavicon(isDarkMode);
            }
        });
    });

    // Start observing the document element for attribute changes
    observer.observe(document.documentElement, { attributes: true });
})();
