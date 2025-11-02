// Token display utilities

/**
 * Create token visualization with color-coded activations
 * @param {Array<string>} tokens - Array of token strings
 * @param {Array<number>} activations - Array of activation values (0-1)
 * @param {number} maxPosition - Index of max activation position
 * @param {boolean} useCustomTooltip - If true, use custom positioned tooltip div; if false, use native title (default: false)
 * @returns {HTMLElement} Container with token visualization
 */
function createTokenVisualization(tokens, activations, maxPosition, useCustomTooltip = false) {
    const tokenContainer = document.createElement('div');
    tokenContainer.className = 'token-container';
    if (useCustomTooltip) {
        tokenContainer.style.position = 'relative';
    }

    tokens.forEach((token, idx) => {
        const span = document.createElement('span');
        span.className = 'token';
        if (useCustomTooltip) {
            span.style.position = 'relative';
        }

        // Handle subword tokens
        // TODO: this is a hack for only some tokenizers
        if (token.startsWith('##')) {
            span.textContent = token.substring(2);
        } else {
            span.textContent = token;
        }

        // Color based on activation
        const activation = activations[idx];
        const normalizedAct = Math.min(Math.max(activation, 0), 1);
        const intensity = Math.floor((1 - normalizedAct) * 255);
        span.style.backgroundColor = `rgb(255, ${intensity}, ${intensity})`;

        // Mark max position
        if (idx === maxPosition) {
            span.style.border = '2px solid blue';
            span.style.fontWeight = 'bold';
        }

        // Add tooltip
        const tooltipText = `${token}: ${activation.toFixed(6)}`;
        if (useCustomTooltip) {
            // Create custom tooltip div
            const tooltip = document.createElement('div');
            tooltip.className = 'token-tooltip';
            tooltip.textContent = tooltipText;
            tooltip.style.display = 'none';
            span.appendChild(tooltip);

            // Show/hide tooltip on hover with dynamic positioning
            span.addEventListener('mouseenter', (e) => {
                const rect = span.getBoundingClientRect();
                tooltip.style.display = 'block';
                tooltip.style.left = rect.left + (rect.width / 2) + 'px';
                tooltip.style.top = rect.top - 5 + 'px';
                tooltip.style.transform = 'translate(-50%, -100%)';
            });
            span.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });
            span.addEventListener('mousemove', (e) => {
                const rect = span.getBoundingClientRect();
                tooltip.style.left = rect.left + (rect.width / 2) + 'px';
                tooltip.style.top = rect.top - 5 + 'px';
            });
        } else {
            // Use native title attribute
            span.title = tooltipText;
        }

        tokenContainer.appendChild(span);

        // Add space after token (unless it's a subword)
        if (!tokens[idx + 1]?.startsWith('##')) {
            tokenContainer.appendChild(document.createTextNode(' '));
        }
    });

    return tokenContainer;
}

// Backwards compatibility alias
function createTokenVisualizationWithTooltip(tokens, activations, maxPosition) {
    return createTokenVisualization(tokens, activations, maxPosition, true);
}