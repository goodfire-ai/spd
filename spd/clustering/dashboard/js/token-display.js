// Token display utilities

/**
 * Tokenizer formatter for BERT-style tokenizers (##subword tokens, spaces between words)
 * @param {string} token - The token string
 * @param {number} idx - Token index
 * @param {Array<string>} tokens - All tokens
 * @returns {{displayText: string, addSpaceAfter: boolean}}
 */
function bertTokenFormatter(token, idx, tokens) {
    const displayText = token.startsWith('##') ? token.substring(2) : token;
    const addSpaceAfter = !tokens[idx + 1]?.startsWith('##');
    return { displayText, addSpaceAfter };
}

/**
 * Tokenizer formatter for GPT2-style tokenizers (spaces encoded as part of token)
 * @param {string} token - The token string
 * @param {number} idx - Token index
 * @param {Array<string>} tokens - All tokens
 * @returns {{displayText: string, addSpaceAfter: boolean}}
 */
function gpt2TokenFormatter(token, idx, tokens) {
    // GPT2 encodes spaces as Ġ character or space at start of token
    // For display, we just show the token as-is, no added spaces
    return { displayText: token, addSpaceAfter: false };
}

/**
 * Create a color scheme function that interpolates from white to a given color
 * @param {string} color - Target color as 'red', 'blue', 'green', or [r, g, b] array
 * @returns {Function} Color scheme function that takes normalizedActivation and returns CSS color
 */
function createColorScheme(color) {
    // Parse color to RGB values
    let r, g, b;
    if (color === 'red') {
        [r, g, b] = [255, 0, 0];
    } else if (color === 'blue') {
        [r, g, b] = [0, 0, 255];
    } else if (color === 'green') {
        [r, g, b] = [0, 255, 0];
    } else if (Array.isArray(color) && color.length === 3) {
        [r, g, b] = color;
    } else {
        throw new Error(`Invalid color: ${color}. Use 'red', 'blue', 'green', or [r, g, b]`);
    }

    return function(normalizedActivation) {
        // Interpolate from white (255, 255, 255) to target color
        const invAct = 1 - normalizedActivation;
        const rVal = Math.floor(255 * invAct + r * normalizedActivation);
        const gVal = Math.floor(255 * invAct + g * normalizedActivation);
        const bVal = Math.floor(255 * invAct + b * normalizedActivation);
        return `rgb(${rVal}, ${gVal}, ${bVal})`;
    };
}

// Common color schemes
const redColorScheme = createColorScheme('red');
const blueColorScheme = createColorScheme('blue');
const greenColorScheme = createColorScheme('green');

/**
 * Create token visualization with color-coded activations and custom tooltips
 * @param {Array<string>} tokens - Array of token strings
 * @param {Array<number>} activations - Array of activation values (0-1)
 * @param {Function} tokenizerFormatter - Function to format tokens for display (default: bertTokenFormatter)
 * @param {Function} colorScheme - Function to map normalized activation to color (default: defaultColorScheme)
 * @returns {HTMLElement} Container with token visualization
 */
function createTokenVisualization(
    tokens,
    activations,
    tokenizerFormatter = bertTokenFormatter,
    colorScheme = redColorScheme
) {
    const tokenContainer = document.createElement('div');
    tokenContainer.className = 'token-container';
    tokenContainer.style.position = 'relative';

    tokens.forEach((token, idx) => {
        const span = document.createElement('span');
        span.className = 'token';

        // Format token using tokenizer-specific logic
        const { displayText, addSpaceAfter } = tokenizerFormatter(token, idx, tokens);
        span.textContent = displayText;

        // Color based on activation
        const activation = activations[idx];
        span.style.backgroundColor = colorScheme(activation);

        // Set tooltip using data-tip attribute
        const tooltipText = `"${token}": ${activation.toFixed(6)}`;
        span.setAttribute('data-tip', tooltipText);

        tokenContainer.appendChild(span);

        // Add space after token based on tokenizer logic
        if (addSpaceAfter) {
            tokenContainer.appendChild(document.createTextNode(' '));
        }
    });

    return tokenContainer;
}