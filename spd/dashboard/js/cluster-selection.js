let componentData = [];
let dashboardData = {};
let dataTable = null;
// TODO: Re-enable explanations feature
// let explanations = {};

// Alpine.js data component for model info
const modelInfoData = {
    data: {},
    hasData: false,

    formatParameters(totalParams) {
        if (!totalParams) return '-';
        if (totalParams >= 1000000) return (totalParams / 1000000).toFixed(1) + 'M';
        if (totalParams >= 1000) return (totalParams / 1000).toFixed(1) + 'K';
        return totalParams.toString();
    },

};

// Custom column renderers
const columnRenderers = {
    componentLabel: function(value, row, col) {
        const container = document.createElement('div');
        container.style.fontFamily = 'monospace';
        container.style.fontSize = '11px';
        container.textContent = value;
        return container;
    },

    moduleName: function(value, row, col) {
        const container = document.createElement('div');
        container.className = 'module-summary';

        // Extract module name from label (format: "module.name:index")
        const parts = value.split('.');
        const displayName = parts.length > 2 ? parts.slice(-2).join('.') : value;

        container.textContent = displayName;
        container.title = value;
        return container;
    },

    activationHistogram: function(value, row, col) {
        const histData = row.histograms?.all_activations;
        if (!histData) {
            return '<span style="color: #999; font-size: 11px;">No data</span>';
        }

        const container = document.createElement('div');
        container.className = 'sparkline-cell';

        // Calculate bin centers for x-axis
        const binCenters = calculateBinCenters(histData.edges);

        const min = row.stats.min;
        const max = row.stats.max;

        // Always use [0, 1] for all_activations
        const xlims = [0, 1];

        // Pass bin centers as x-values and counts as y-values
        const svg = sparkbars(binCenters, histData.counts, {
            width: CONFIG.visualization.sparklineWidth,
            height: CONFIG.visualization.sparklineHeight,
            color: '#4169E1',
            shading: true,
            lineWidth: 0,
            markers: '',
            margin: 2,
            xlims: xlims,
            ylims: [0, null],
            logScale: true,
            xAxis: {line: true, ticks: true, label_margin: 10},
            yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin}
        });

        container.innerHTML = svg;

        const mean = row.stats.mean;
        const median = row.stats.median;
        const totalCount = histData.counts.reduce((a, b) => a + b, 0);

        container.title = `All Activations Histogram (n=${totalCount})\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

        return container;
    },

    maxActivationDistribution: function(value, row, col) {
        const histData = row.histograms?.max_per_sample;
        if (!histData) {
            return '<span style="color: #999; font-size: 11px;">No data</span>';
        }

        const container = document.createElement('div');
        container.className = 'sparkline-cell';

        // Calculate bin centers for x-axis
        const binCenters = calculateBinCenters(histData.edges);

        const min = histData.edges[0];
        const max = histData.edges[histData.edges.length - 1];

        // Always use [0, 1] for max_per_sample
        const xlims = [0, 1];

        // Pass bin centers as x-values and counts as y-values
        const svg = sparkbars(binCenters, histData.counts, {
            width: CONFIG.visualization.sparklineWidth,
            height: CONFIG.visualization.sparklineHeight,
            color: '#DC143C',
            shading: true,
            lineWidth: 0,
            markers: '',
            margin: 2,
            xlims: xlims,
            ylims: [0, null],
            logScale: true,
            xAxis: {line: true, ticks: true, label_margin: 10},
            yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin}
        });

        container.innerHTML = svg;

        const mean = calculateHistogramMean(histData);
        const median = calculateHistogramMedian(histData);
        const totalCount = histData.counts.reduce((a, b) => a + b, 0);

        container.title = `Max Activation Distribution (n=${totalCount} samples)\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

        return container;
    },

    componentLink: function(value, row, col) {
        return `<a href="component.html?label=${encodeURIComponent(row.label)}">View →</a>`;
    },

    // TODO: Re-enable explanations feature
    // explanation: function(value, row, col) {
    //     if (!value) {
    //         return '<span style="color: #999; font-style: italic;">—</span>';
    //     }
    //     // Truncate long explanations
    //     const maxLength = 60;
    //     if (value.length > maxLength) {
    //         const truncated = value.substring(0, maxLength) + '...';
    //         const span = document.createElement('span');
    //         span.textContent = truncated;
    //         span.title = value;  // Show full text on hover
    //         return span;
    //     }
    //     return value;
    // },

    tokenEntropy: function(value, row, col) {
        const tokenStats = row.stats.token_activations;
        if (!tokenStats) {
            return '<span style="color: #999; font-size: 11px;">N/A</span>';
        }
        return tokenStats.entropy.toFixed(2);
    },

    tokenConcentration: function(value, row, col) {
        const tokenStats = row.stats.token_activations;
        if (!tokenStats) {
            return '<span style="color: #999; font-size: 11px;">N/A</span>';
        }
        return (tokenStats.concentration_ratio * 100).toFixed(1) + '%';
    },

    topToken: function(value, row, col) {
        const tokenStats = row.stats.token_activations;
        if (!tokenStats || !tokenStats.top_tokens || tokenStats.top_tokens.length === 0) {
            return '<span style="color: #999; font-size: 11px;">N/A</span>';
        }

        const container = document.createElement('div');
        container.style.fontFamily = 'monospace';
        container.style.fontSize = '11px';
        container.style.lineHeight = '1.4';

        const topN = Math.min(5, tokenStats.top_tokens.length);
        const maxPercentage = tokenStats.top_tokens.length > 0
            ? ((tokenStats.top_tokens[0].count / tokenStats.total_activations) * 100)
            : 0;

        for (let i = 0; i < topN; i++) {
            const token = tokenStats.top_tokens[i];
            const tokenDisplay = token.token.replace(/ /g, '·').replace(/\n/g, '↵');
            const percentageValue = ((token.count / tokenStats.total_activations) * 100);
            const percentage = percentageValue.toFixed(1);

            // Color based on percentage (normalized by max percentage)
            const normalizedPct = maxPercentage > 0 ? percentageValue / maxPercentage : 0;
            const intensity = Math.floor((1 - normalizedPct) * 255);
            const bgColor = `rgb(255, ${intensity}, ${intensity})`;

            const line = document.createElement('div');
            line.style.display = 'flex';
            line.style.justifyContent = 'space-between';
            line.style.gap = '8px';

            const tokenSpan = document.createElement('span');
            tokenSpan.innerHTML = `<code class="token-display">${tokenDisplay}</code>`;
            tokenSpan.style.textAlign = 'left';

            const pctSpan = document.createElement('span');
            pctSpan.textContent = `${percentage}%`;
            pctSpan.style.textAlign = 'right';
            pctSpan.style.backgroundColor = bgColor;
            pctSpan.style.padding = '2px 4px';
            pctSpan.style.borderRadius = '2px';

            line.appendChild(tokenSpan);
            line.appendChild(pctSpan);
            container.appendChild(line);
        }

        return container;
    },

    tokensGivenActive: function(value, row, col) {
        return createTokensGivenActiveCell(
            row.top_tokens_given_active,
            CONFIG.tokenStats.displayTopN
        );
    },

    activeGivenTokens: function(value, row, col) {
        return createActiveGivenTokensCell(
            row.top_active_given_tokens,
            CONFIG.tokenStats.displayTopN
        );
    },

    // Generic histogram renderer for any histogram in row.histograms
    genericHistogram: function(histKey, color, title) {
        return function(value, row, col) {
            const histData = row.histograms?.[histKey];
            if (!histData || !histData.counts) {
                return '<span style="color: #999; font-size: 11px;">No data</span>';
            }

            const container = document.createElement('div');
            container.className = 'sparkline-cell';

            // Calculate bin centers for x-axis
            const binCenters = calculateBinCenters(histData.edges);

            // Calculate statistics of underlying data
            const min = histData.edges[0];
            const max = histData.edges[histData.edges.length - 1];

            // Force [0, 1] xlims for activation histograms, otherwise auto-scale
            const xlims = (histKey === 'all_activations' || histKey === 'max_per_sample') ? [0, 1] : null;

            // Pass bin centers as x-values and counts as y-values
            const svg = sparkbars(binCenters, histData.counts, {
                width: CONFIG.visualization.sparklineWidth,
                height: CONFIG.visualization.sparklineHeight,
                color: color,
                shading: true,
                lineWidth: 0,
                markers: '',
                margin: 2,
                xlims: xlims,
                ylims: [0, null],
                logScale: true,
                xAxis: {line: true, ticks: true, label_margin: 10},
                yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin}
            });

            container.innerHTML = svg;

            const mean = calculateHistogramMean(histData);
            const median = calculateHistogramMedian(histData);
            const totalCount = histData.counts.reduce((a, b) => a + b, 0);

            container.title = `${title} (n=${totalCount})\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

            return container;
        };
    }
};

// ============================================================================
// Helper Functions for Filtering and Sorting
// ============================================================================

/**
 * Create a filter function for module name (string) with wildcards
 * @param {string} filterValue - The filter pattern (supports * wildcards)
 * @returns {Function|null} Filter function or null if invalid
 */
function createModuleFilter(filterValue) {
    if (!filterValue || !filterValue.trim()) return null;

    const pattern = filterValue.toLowerCase().trim();
    const regex = pattern.includes('*')
        ? new RegExp('^' + pattern.replace(/\*/g, '.*') + '$')
        : null;

    return (cellValue) => {
        const moduleLower = cellValue.toLowerCase();
        return regex
            ? regex.test(moduleLower)
            : moduleLower.includes(pattern);
    };
}

/**
 * Parse extended histogram filter syntax (e.g., "mean>0.5", "max<10", "mean>0.5, max<10")
 * @param {string} filterValue - The filter string (can be comma-separated for multiple conditions)
 * @returns {Array|null} Array of parsed filters [{ statType, operator, value }] or null if plain numeric
 */
function parseHistogramFilter(filterValue) {
    const trimmed = filterValue.trim();
    if (!trimmed) return null;

    // Split by comma to support multiple conditions
    const conditions = trimmed.split(',').map(c => c.trim());
    const parsedConditions = [];

    for (const condition of conditions) {
        // Match pattern: statType operator value (e.g., "mean>0.5", "median<=0.2")
        const match = condition.match(/^(mean|median|max|min|range|sum)\s*(==|!=|>=|<=|>|<)\s*(-?\d+\.?\d*)$/i);

        if (match) {
            parsedConditions.push({
                statType: match[1].toLowerCase(),
                operator: match[2],
                value: parseFloat(match[3])
            });
        } else {
            // If any condition doesn't match, return null to use default filter
            return null;
        }
    }

    // Return array of conditions, or null if none were found
    return parsedConditions.length > 0 ? parsedConditions : null;
}

/**
 * Create a filter function for histogram columns with extended syntax
 * Supports multiple comma-separated conditions (AND logic)
 * @param {string} histKey - The histogram key in row.histograms
 * @param {string} filterValue - The filter string (e.g., "mean>0.5, max<10")
 * @returns {Function|null} Filter function or null to use default
 */
function createHistogramFilter(histKey, filterValue) {
    const parsedConditions = parseHistogramFilter(filterValue);

    if (!parsedConditions) {
        // Return null to let default numeric filter handle it
        // Default will filter on the sort value (mean by default)
        return null;
    }

    return (cellValue, row) => {
        // All conditions must be satisfied (AND logic)
        for (const condition of parsedConditions) {
            const { statType, operator, value } = condition;
            const histData = row.histograms?.[histKey];

            if (!histData || !histData.counts || !histData.edges) return false;

            // Calculate the requested statistic
            let statValue;
            switch (statType) {
                case 'mean':
                    // Use precomputed stats if available
                    if (histKey === 'all_activations' && row.stats.mean !== undefined) {
                        statValue = row.stats.mean;
                    } else {
                        statValue = calculateHistogramMean(histData);
                    }
                    break;
                case 'median':
                    if (histKey === 'all_activations' && row.stats.median !== undefined) {
                        statValue = row.stats.median;
                    } else {
                        statValue = calculateHistogramMedian(histData);
                    }
                    break;
                case 'max':
                    if (histKey === 'all_activations' && row.stats.max !== undefined) {
                        statValue = row.stats.max;
                    } else {
                        statValue = histData.edges[histData.edges.length - 1];
                    }
                    break;
                case 'min':
                    if (histKey === 'all_activations' && row.stats.min !== undefined) {
                        statValue = row.stats.min;
                    } else {
                        statValue = histData.edges[0];
                    }
                    break;
                case 'range':
                    statValue = histData.edges[histData.edges.length - 1] - histData.edges[0];
                    break;
                case 'sum':
                    statValue = histData.counts.reduce((a, b) => a + b, 0);
                    break;
                default:
                    return false;
            }

            if (statValue === null || statValue === undefined) return false;

            let conditionMet = false;
            switch (operator) {
                case '==': conditionMet = Math.abs(statValue - value) < 0.0001; break;
                case '!=': conditionMet = Math.abs(statValue - value) >= 0.0001; break;
                case '>': conditionMet = statValue > value; break;
                case '<': conditionMet = statValue < value; break;
                case '>=': conditionMet = statValue >= value; break;
                case '<=': conditionMet = statValue <= value; break;
                default: conditionMet = false;
            }

            // If any condition fails, return false
            if (!conditionMet) return false;
        }

        // All conditions passed
        return true;
    };
}

/**
 * Get the top token string for sorting
 * @param {object} value - Cell value (stats object)
 * @param {object} row - The data row
 * @returns {string} The top token string for sorting
 */
function sortTopToken(value, row) {
    const tokenStats = row.stats.token_activations;
    if (!tokenStats || !tokenStats.top_tokens || tokenStats.top_tokens.length === 0) {
        return '';
    }
    return tokenStats.top_tokens[0].token.toLowerCase();
}

/**
 * Create a filter function for top tokens
 * @param {string} filterValue - The filter string
 * @returns {Function|null} Filter function or null if invalid
 */
function createTopTokenFilter(filterValue) {
    if (!filterValue || !filterValue.trim()) return null;

    const pattern = filterValue.toLowerCase().trim();

    return (cellValue, row) => {
        const tokenStats = row.stats.token_activations;
        if (!tokenStats || !tokenStats.top_tokens) return false;

        // Search in top 10 tokens
        const topN = Math.min(10, tokenStats.top_tokens.length);
        for (let i = 0; i < topN; i++) {
            const token = tokenStats.top_tokens[i].token.toLowerCase();
            if (token.includes(pattern)) {
                return true;
            }
        }
        return false;
    };
}

/**
 * Create filter function for token statistics columns
 * @param {string} tokenStatsKey - The key in row object (e.g., 'top_tokens_given_active')
 * @returns {Function} Filter function factory
 */
function createTokenStatsFilter(tokenStatsKey) {
    return function(filterValue) {
        if (!filterValue || !filterValue.trim()) return null;

        const pattern = filterValue.toLowerCase().trim();

        return (cellValue, row) => {
            const tokenStats = row[tokenStatsKey];
            if (!tokenStats || tokenStats.length === 0) return false;

            // Search through all tokens in the array
            for (const stat of tokenStats) {
                if (stat.token.toLowerCase().includes(pattern)) {
                    return true;
                }
            }
            return false;
        };
    };
}

/**
 * Create a filter function for numeric comparisons with operators
 * @param {string} filterValue - The filter string (e.g., ">2.5", "<=0.8")
 * @param {Function} valueExtractor - Function to extract numeric value from cellValue
 * @returns {Function|null} Filter function or null if invalid
 */
function createNumericFilter(filterValue, valueExtractor) {
    if (!filterValue || !filterValue.trim()) return null;

    const trimmed = filterValue.trim();

    // Match pattern: operator value (e.g., ">2.5", "<=0.8")
    const match = trimmed.match(/^(==|!=|>=|<=|>|<)\s*(-?\d+\.?\d*)$/);

    if (!match) {
        // Try plain number (defaults to ==)
        const plainNum = parseFloat(trimmed);
        if (!isNaN(plainNum)) {
            return (cellValue, row) => {
                const value = valueExtractor(cellValue);
                if (value === null || value === undefined) return false;
                return Math.abs(value - plainNum) < 0.0001;
            };
        }
        return null;
    }

    const operator = match[1];
    const targetValue = parseFloat(match[2]);

    return (cellValue, row) => {
        const value = valueExtractor(cellValue);
        if (value === null || value === undefined) return false;

        switch (operator) {
            case '==': return Math.abs(value - targetValue) < 0.0001;
            case '!=': return Math.abs(value - targetValue) >= 0.0001;
            case '>': return value > targetValue;
            case '<': return value < targetValue;
            case '>=': return value >= targetValue;
            case '<=': return value <= targetValue;
            default: return false;
        }
    };
}

async function processComponentData() {
    const tableData = [];

    for (const component of componentData) {
        // Extract module and index from label (format: "module.name:index")
        const labelParts = component.label.split(':');
        const moduleName = labelParts[0];
        const componentIndex = labelParts.length > 1 ? parseInt(labelParts[1]) : 0;

        tableData.push({
            label: component.label,
            module: moduleName,
            index: componentIndex,
            stats: component.stats,
            histograms: component.histograms,
            embedding: component.embedding,
            top_tokens_given_active: component.top_tokens_given_active,
            top_active_given_tokens: component.top_active_given_tokens
            // TODO: Re-enable explanations feature
            // explanation: explanation
        });
    }

    return tableData;
}

async function loadData() {
    // Load data via ZANJ
    const loader = new ZanjLoader(CONFIG.data.dataDir);
    const data = await loader.read();

    // Extract data
    dashboardData = data;
    componentData = await data.components;

    // TODO: Re-enable explanations feature
    // Load explanations separately (not part of ZANJ)
    // explanations = await loadJSONL(CONFIG.getDataPath('explanations'), 'component_label').catch(() => ({}));

    const tableData = await processComponentData();

    // Discover histogram stats from first component
    const firstComponent = componentData[0];
    const histogramStats = [];
    if (firstComponent && firstComponent.histograms) {
        histogramStats.push(...Object.keys(firstComponent.histograms));
    }

    // Base columns
    const columns = [
        {
            key: 'module',
            label: 'Module',
            type: 'string',
            width: '150px',
            renderer: columnRenderers.moduleName,
            filterFunction: (filterValue) => createModuleFilter(filterValue),
            filterTooltip: 'Filter by module name. Use * for wildcards. Example: *mlp*, *attn*'
        },
        {
            key: 'index',
            label: 'Index',
            type: 'number',
            width: '80px',
            align: 'right',
            renderer: (value) => value.toString()
        }
    ];

    // Add histogram columns dynamically
    const statColors = {
        'all_activations': '#4169E1',
        'max_per_sample': '#DC143C',
        'mean_per_sample': '#228B22'
    };

    histogramStats.forEach(histKey => {
        const color = statColors[histKey] || '#808080';
        const label = histKey.replace(/-/g, ' ').replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

        columns.push({
            id: 'histogram_' + histKey,
            key: 'histograms',
            label: label,
            type: 'number',
            width: '200px',
            align: 'center',
            renderer: columnRenderers.genericHistogram(histKey, color, label),
            sortFunction: (value, row) => {
                const histData = row.histograms?.[histKey];
                if (!histData || !histData.counts || !histData.edges) return -Infinity;
                // For all_activations, use precomputed mean
                if (histKey === 'all_activations' && row.stats.mean !== undefined) {
                    return row.stats.mean;
                }
                // Otherwise calculate mean from histogram
                return calculateHistogramMean(histData);
            },
            filterFunction: (filterValue) => createHistogramFilter(histKey, filterValue),
            filterTooltip: 'Filter by statistics. Use: mean>0.5, median<0.2, max>=1.0, min>-0.1, range<5, sum>100. Combine with commas (e.g., mean>0.5, max<10)'
        });
    });

    // Token statistics columns
    columns.push({
        id: 'tokens_given_active',
        key: 'top_tokens_given_active',
        label: 'P(token|active)',
        type: 'string',
        width: '150px',
        align: 'left',
        renderer: columnRenderers.tokensGivenActive
    });

    columns.push({
        id: 'active_given_tokens',
        key: 'top_active_given_tokens',
        label: 'P(active|token)',
        type: 'string',
        width: '150px',
        align: 'left',
        renderer: columnRenderers.activeGivenTokens
    });
    // columns.push({
    //     id: 'token_entropy',
    //     key: 'stats',
    //     label: 'Token Entropy',
    //     type: 'number',
    //     width: '60px',
    //     align: 'right',
    //     renderer: columnRenderers.tokenEntropy,
    //     sortFunction: (value, row) => {
    //         const tokenStats = row.stats.token_activations;
    //         return tokenStats ? tokenStats.entropy : -Infinity;
    //     },
    //     filterFunction: (filterValue) => createNumericFilter(filterValue, (stats) => {
    //         const tokenStats = stats?.token_activations;
    //         return tokenStats ? tokenStats.entropy : null;
    //     }),
    //     filterTooltip: 'Filter by entropy. Use operators: >, <, >=, <=, ==, != (e.g., >2.5)'
    // });

    // columns.push({
    //     id: 'token_concentration',
    //     key: 'stats',
    //     label: 'Token Conc.',
    //     type: 'number',
    //     width: '60px',
    //     align: 'right',
    //     renderer: columnRenderers.tokenConcentration,
    //     sortFunction: (value, row) => {
    //         const tokenStats = row.stats.token_activations;
    //         return tokenStats ? tokenStats.concentration_ratio : -Infinity;
    //     },
    //     filterFunction: (filterValue) => createNumericFilter(filterValue, (stats) => {
    //         const tokenStats = stats?.token_activations;
    //         return tokenStats ? tokenStats.concentration_ratio : null;
    //     }),
    //     filterTooltip: 'Filter by concentration (0-1). Use operators: >, <, >=, <=, ==, != (e.g., >0.5)'
    // });

    // TODO: Re-enable explanations feature
    // Explanation column
    // columns.push({
    //     key: 'explanation',
    //     label: 'Explanation',
    //     type: 'string',
    //     width: '200px',
    //     align: 'left',
    //     renderer: columnRenderers.explanation,
    //     filterTooltip: 'Filter by explanation text (case-insensitive substring match)'
    // });

    // Actions column
    columns.push({
        key: 'label',
        label: 'Actions',
        type: 'string',
        width: '20px',
        align: 'center',
        renderer: columnRenderers.componentLink,
        filterable: false
    });

    const tableConfig = {
        data: tableData,
        columns: columns,
        pageSize: CONFIG.indexPage.pageSize,
        pageSizeOptions: CONFIG.indexPage.pageSizeOptions,
        showFilters: CONFIG.indexPage.showFilters
    };

    dataTable = new DataTable('#componentTableContainer', tableConfig);

    const loading = document.getElementById('loading');
    if (!loading) {
        const msg = 'Fatal error: loading element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    loading.style.display = 'none';
}

document.addEventListener('DOMContentLoaded', async () => {
    await initConfig();

    // Check if Alpine.js loaded
    if (typeof Alpine === 'undefined') {
        const msg = 'Fatal error: Alpine.js failed to load. Check your internet connection or CDN.';
        NOTIF.error(msg, null, null);
        console.error(msg);
    }

    // Load component data and render table (includes dashboard metadata from ZANJ)
    await loadData();

    // Populate Alpine.js component with loaded dashboard metadata
    const modelInfoEl = document.getElementById('modelInfo');
    if (modelInfoEl && Alpine.$data(modelInfoEl)) {
        Alpine.$data(modelInfoEl).data = dashboardData;
        Alpine.$data(modelInfoEl).hasData = Object.keys(dashboardData).length > 0;
    }
});
