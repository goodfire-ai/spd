// token-stats.js - Token statistics rendering utilities

/**
 * Create a compact display of P(token|active) for a table cell
 * @param {Array} tokensGivenActive - Array of TokenStat objects for P(token|active)
 * @param {number} topN - Number of top tokens to display
 * @returns {HTMLElement} Container with formatted token statistics
 */
function createTokensGivenActiveCell(tokensGivenActive, topN) {
    if (!tokensGivenActive || tokensGivenActive.length === 0) {
        const span = document.createElement('span');
        span.style.color = '#999';
        span.style.fontSize = '11px';
        span.textContent = 'N/A';
        return span;
    }

    const container = document.createElement('div');
    container.style.fontFamily = 'monospace';
    container.style.fontSize = '10px';
    container.style.lineHeight = '1.5';

    appendTokenStatsSection(
        container,
        tokensGivenActive.slice(0, topN),
        (stat) => stat.p_token_given_active
    );

    return container;
}

/**
 * Create a compact display of P(active|token) for a table cell
 * @param {Array} activeGivenTokens - Array of TokenStat objects for P(active|token)
 * @param {number} topN - Number of top tokens to display
 * @returns {HTMLElement} Container with formatted token statistics
 */
function createActiveGivenTokensCell(activeGivenTokens, topN) {
    if (!activeGivenTokens || activeGivenTokens.length === 0) {
        const span = document.createElement('span');
        span.style.color = '#999';
        span.style.fontSize = '11px';
        span.textContent = 'N/A';
        return span;
    }

    const container = document.createElement('div');
    container.style.fontFamily = 'monospace';
    container.style.fontSize = '10px';
    container.style.lineHeight = '1.5';

    appendTokenStatsSection(
        container,
        activeGivenTokens.slice(0, topN),
        (stat) => stat.p_active_given_token
    );

    return container;
}

/**
 * Append a token statistics section to a container (no header)
 * @param {HTMLElement} container - Container to append to
 * @param {Array} stats - Array of TokenStat objects
 * @param {Function} getProbability - Function to extract probability from TokenStat
 */
function appendTokenStatsSection(container, stats, getProbability) {
    if (stats.length === 0) return;

    for (const stat of stats) {
        const line = createTokenStatLine(stat.token, getProbability(stat));
        container.appendChild(line);
    }
}

/**
 * Create a single token stat line with token and percentage
 * @param {string} token - Token string
 * @param {number} probability - Probability value (0-1)
 * @returns {HTMLElement} Line element
 */
function createTokenStatLine(token, probability) {
    const tokenDisplay = formatTokenDisplay(token);
    const pct = (probability * 100).toFixed(1);

    const line = document.createElement('div');
    line.style.display = 'flex';
    line.style.justifyContent = 'space-between';
    line.style.gap = '6px';

    const tokenSpan = document.createElement('span');
    tokenSpan.innerHTML = `<code class="token-display">${tokenDisplay}</code>`;
    tokenSpan.style.textAlign = 'left';
    tokenSpan.style.flex = '1';
    tokenSpan.style.overflow = 'hidden';
    tokenSpan.style.textOverflow = 'ellipsis';
    tokenSpan.style.whiteSpace = 'nowrap';

    const pctSpan = document.createElement('span');
    pctSpan.textContent = `${pct}%`;
    pctSpan.style.textAlign = 'right';
    pctSpan.style.minWidth = '40px';
    pctSpan.style.padding = '1px 4px';
    pctSpan.style.borderRadius = '2px';

    // Add background color based on probability (higher = darker)
    const bgColor = getProbabilityColor(probability);
    pctSpan.style.backgroundColor = bgColor;

    // Use white text for darker backgrounds
    if (probability > 0.5) {
        pctSpan.style.color = 'white';
    }

    line.appendChild(tokenSpan);
    line.appendChild(pctSpan);

    return line;
}

/**
 * Get background color for probability value (gradient from light to dark)
 * @param {number} probability - Probability value (0-1)
 * @returns {string} RGB color string
 */
function getProbabilityColor(probability) {
    // Clamp probability to [0, 1]
    const p = Math.max(0, Math.min(1, probability));

    // Blue gradient: light (#E3F2FD) to dark (#1976D2)
    // RGB: (227, 242, 253) to (25, 118, 210)
    const r = Math.round(227 + (25 - 227) * p);
    const g = Math.round(242 + (118 - 242) * p);
    const b = Math.round(253 + (210 - 253) * p);

    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Format token for display (replace spaces and newlines)
 * @param {string} token - Raw token string
 * @returns {string} Formatted token string
 */
function formatTokenDisplay(token) {
    return token.replace(/ /g, '·').replace(/\n/g, '↵');
}
