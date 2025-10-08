let clusterData = {};
let modelInfo = {};
let dataTable = null;
let explanations = {};

// Alpine.js data component for model info
const modelInfoData = {
    data: {},
    hasData: false,

    async loadData() {
        try {
            const response = await fetch(CONFIG.getDataPath('modelInfo'));
            this.data = await response.json();
            this.hasData = Object.keys(this.data).length > 0;

            // Also populate global modelInfo for DataTable renderers
            modelInfo = this.data;

            console.log('Model info loaded:', this.hasData, Object.keys(this.data));
        } catch (error) {
            console.error('Failed to load model info:', error);
            this.hasData = false;
        }
    },

    formatParameters(totalParams) {
        if (!totalParams) return '-';
        if (totalParams >= 1000000) return (totalParams / 1000000).toFixed(1) + 'M';
        if (totalParams >= 1000) return (totalParams / 1000).toFixed(1) + 'K';
        return totalParams.toString();
    },

    formatWandBLink(path) {
        if (!path) return '-';

        // Remove "wandb:" prefix if present
        const cleanPath = path.replace(/^wandb:/, '');

        // Convert to WandB URL
        const url = `https://wandb.ai/${cleanPath}`;

        // Show shortened path in link text
        const displayText = cleanPath.length > 60
            ? cleanPath.substring(0, 57) + '...'
            : cleanPath;

        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${displayText}</a>`;
    }
};

// Custom column renderers
const columnRenderers = {
    modelView: function(value, row, col) {
        const container = document.createElement('div');
        container.className = 'modelview-cell';

        renderModelView(container, row.clusterHash, clusterData, modelInfo, CONFIG.visualization.colormap, CONFIG.visualization.modelViewCellSizeTable);

        return container;
    },

    modulesSummary: function(value, row, col) {
        const modules = row.modules;
        const container = document.createElement('div');
        container.className = 'module-summary';

        if (modules.length === 1) {
            const parts = modules[0].split('.');
            container.textContent = parts.length > 2 ? parts.slice(-2).join('.') : modules[0];
        } else if (modules.length <= 3) {
            container.textContent = modules.map(m => {
                const parts = m.split('.');
                return parts.length > 2 ? parts.slice(-2).join('.') : m;
            }).join(', ');
        } else {
            container.textContent = `${modules.length} modules`;
        }

        container.title = modules.join('\n');
        return container;
    },

    activationHistogram: function(value, row, col) {
        const histData = row.stats.all_activations;
        if (!histData) {
            return '<span style="color: #999; font-size: 11px;">No data</span>';
        }

        const container = document.createElement('div');
        container.className = 'sparkline-cell';

        // Calculate bin centers for x-axis
        const binCenters = calculateBinCenters(histData.bin_edges);

        const min = row.stats.min_activation;
        const max = row.stats.max_activation;

        // Set x-axis limits to [0, 1] if data is in that range
        const xlims = (min >= 0 && max <= 1) ? [0, 1] : null;

        // Pass bin centers as x-values and counts as y-values
        const svg = sparkbars(binCenters, histData.bin_counts, {
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

        const mean = row.stats.mean_activation;
        const median = calculateHistogramMedian(histData);
        const n = row.stats.n_tokens;

        container.title = `All Activations Histogram (n=${n})\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

        return container;
    },

    maxActivationDistribution: function(value, row, col) {
        const histData = row.stats['max_activation-max-16'];
        if (!histData) {
            return '<span style="color: #999; font-size: 11px;">No data</span>';
        }

        const container = document.createElement('div');
        container.className = 'sparkline-cell';

        // Calculate bin centers for x-axis
        const binCenters = calculateBinCenters(histData.bin_edges);

        const min = histData.bin_edges[0];
        const max = histData.bin_edges[histData.bin_edges.length - 1];

        // Set x-axis limits to [0, 1] if data is in that range
        const xlims = (min >= 0 && max <= 1) ? [0, 1] : null;

        // Pass bin centers as x-values and counts as y-values
        const svg = sparkbars(binCenters, histData.bin_counts, {
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

        const n = row.stats.n_samples;
        const mean = calculateHistogramMean(histData);
        const median = calculateHistogramMedian(histData);

        container.title = `Max Activation Distribution (n=${n} samples)\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

        return container;
    },

    clusterLink: function(value, row, col) {
        return `<a href="cluster.html?id=${row.clusterHash}">View →</a>`;
    },

    explanation: function(value, row, col) {
        if (!value) {
            return '<span style="color: #999; font-style: italic;">—</span>';
        }
        // Truncate long explanations
        const maxLength = 60;
        if (value.length > maxLength) {
            const truncated = value.substring(0, maxLength) + '...';
            const span = document.createElement('span');
            span.textContent = truncated;
            span.title = value;  // Show full text on hover
            return span;
        }
        return value;
    },

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

    // Generic histogram renderer for any BinnedData stat
    genericHistogram: function(statKey, color, title) {
        return function(value, row, col) {
            const histData = row.stats[statKey];
            if (!histData || !histData.bin_counts) {
                return '<span style="color: #999; font-size: 11px;">No data</span>';
            }

            const container = document.createElement('div');
            container.className = 'sparkline-cell';

            // Calculate bin centers for x-axis
            const binCenters = calculateBinCenters(histData.bin_edges);

            // Calculate statistics of underlying data
            const min = histData.bin_edges[0];
            const max = histData.bin_edges[histData.bin_edges.length - 1];

            // Set x-axis limits to [0, 1] if data is in that range
            const xlims = (min >= 0 && max <= 1) ? [0, 1] : null;

            // Pass bin centers as x-values and counts as y-values
            const svg = sparkbars(binCenters, histData.bin_counts, {
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
            const totalCount = histData.bin_counts.reduce((a, b) => a + b, 0);

            container.title = `${title} (n=${totalCount})\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

            return container;
        };
    }
};

// ============================================================================
// Helper Functions for Filtering and Sorting
// ============================================================================

/**
 * Create a filter function for module arrays that supports wildcards, multiple patterns, and negation
 * @param {string} filterValue - The filter pattern (supports * wildcards, , for OR, & for AND, @ for all-match, ! for negation)
 * @returns {Function|null} Filter function or null if invalid
 */
function createModuleFilter(filterValue) {
    if (!filterValue || !filterValue.trim()) return null;

    // Split by comma for OR groups
    const orGroups = filterValue.split(',').map(g => g.trim()).filter(g => g);

    // Parse each OR group (which may contain & for AND)
    const parsedOrGroups = orGroups.map(group => {
        // Split by & for AND conditions within this OR group
        const andConditions = group.split('&').map(c => c.trim()).filter(c => c);

        return andConditions.map(condition => {
            let mode = 'some'; // default: at least one module matches
            let negate = false;
            let pattern = condition.toLowerCase();

            // Check for @ prefix (all modules must match)
            if (pattern.startsWith('@')) {
                mode = 'every';
                pattern = pattern.substring(1);
            }
            // Check for ! prefix (no modules can match)
            else if (pattern.startsWith('!')) {
                negate = true;
                pattern = pattern.substring(1);
            }

            const regex = pattern.includes('*')
                ? new RegExp('^' + pattern.replace(/\*/g, '.*') + '$')
                : null;

            return { mode, negate, pattern, regex };
        });
    });

    return (cellValue) => {
        // cellValue is the modules array
        if (!Array.isArray(cellValue)) return false;

        // OR logic across groups
        return parsedOrGroups.some(andGroup => {
            // AND logic within group
            return andGroup.every(condition => {
                const matchFn = (module) => {
                    const moduleLower = module.toLowerCase();
                    return condition.regex
                        ? condition.regex.test(moduleLower)
                        : moduleLower.includes(condition.pattern);
                };

                if (condition.mode === 'every') {
                    // ALL modules must match
                    const result = cellValue.every(matchFn);
                    return condition.negate ? !result : result;
                } else {
                    // At least ONE module must match (or none if negated)
                    const result = cellValue.some(matchFn);
                    return condition.negate ? !result : result;
                }
            });
        });
    };
}

/**
 * Sort function for module arrays
 * Primary: number of modules (ascending)
 * Secondary: alphabetically by first module name
 * @param {Array} modules - Array of module names
 * @returns {string} Sortable string representation
 */
function sortModules(modules) {
    if (!Array.isArray(modules) || modules.length === 0) return '';

    // Pad module count for proper numeric sorting, then add first module name
    const count = modules.length.toString().padStart(5, '0');
    const firstName = modules[0].toLowerCase();
    return `${count}_${firstName}`;
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
 * @param {string} statKey - The statistics key
 * @param {string} filterValue - The filter string (e.g., "mean>0.5, max<10")
 * @returns {Function|null} Filter function or null to use default
 */
function createHistogramFilter(statKey, filterValue) {
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
            const histData = row.stats[statKey];

            if (!histData || !histData.bin_counts || !histData.bin_edges) return false;

            // Calculate the requested statistic
            let statValue;
            switch (statType) {
                case 'mean':
                    // For all_activations, use precomputed mean
                    if (statKey === 'all_activations' && row.stats.mean_activation !== undefined) {
                        statValue = row.stats.mean_activation;
                    } else {
                        statValue = calculateHistogramMean(histData);
                    }
                    break;
                case 'median':
                    statValue = calculateHistogramMedian(histData);
                    break;
                case 'max':
                    statValue = histData.bin_edges[histData.bin_edges.length - 1];
                    break;
                case 'min':
                    statValue = histData.bin_edges[0];
                    break;
                case 'range':
                    statValue = histData.bin_edges[histData.bin_edges.length - 1] - histData.bin_edges[0];
                    break;
                case 'sum':
                    statValue = histData.bin_counts.reduce((a, b) => a + b, 0);
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

function processClusterData() {
    const tableData = [];

    for (const [clusterHash, cluster] of Object.entries(clusterData)) {
        const modules = new Set();
        cluster.components.forEach(comp => {
            modules.add(comp.module);
        });

        const stats = cluster.stats;

        // Extract cluster ID from hash (format: "runid-iteration-clusteridx")
        const parts = clusterHash.split('-');
        const clusterId = parseInt(parts[parts.length - 1]);

        // Get explanation for this cluster
        const explanationData = explanations[clusterHash];
        const explanation = explanationData ? explanationData.explanation : null;

        tableData.push({
            id: clusterId,
            clusterHash: clusterHash,
            componentCount: cluster.components.length,
            modules: Array.from(modules),
            stats: stats,
            explanation: explanation
        });
    }

    return tableData;
}

async function loadData() {
    // Load cluster data (model info is handled by Alpine.js)
    const clusters = await loadJSONL(CONFIG.getDataPath('clusters'), 'cluster_hash');

    clusterData = clusters;

    // Load explanations (non-critical, don't fail if missing)
    explanations = await loadJSONL(CONFIG.getDataPath('explanations'), 'cluster_id').catch(() => ({}));

    const tableData = processClusterData();

    // Discover histogram stats from first cluster
    const firstCluster = Object.values(clusterData)[0];
    const histogramStats = [];
    if (firstCluster && firstCluster.stats) {
        for (const [key, value] of Object.entries(firstCluster.stats)) {
            if (value && typeof value === 'object' && 'bin_counts' in value && 'bin_edges' in value) {
                histogramStats.push(key);
            }
        }
    }

    // Base columns
    const columns = [
        {
            key: 'id',
            label: 'ID',
            type: 'number',
            width: '10px',
            align: 'center'
        },
        {
            key: 'componentCount',
            label: 'Comps',
            type: 'number',
            width: '10px',
            align: 'right'
        },
        {
            key: 'modules',
            label: 'Model View',
            type: 'string',
            width: '21px',
            align: 'center',
            renderer: columnRenderers.modelView,
            sortFunction: (modules) => sortModules(modules),
            filterFunction: (filterValue) => createModuleFilter(filterValue),
            filterTooltip: 'Filter by module. Separate with , (OR) or & (AND). Use * for wildcards. Prefix @ for all-match, ! to exclude. Examples: *mlp*,*attn* (OR), *mlp*&*attn* (AND), @*proj* (all), !*o_proj* (exclude)'
        },
        {
            key: 'modules',
            label: 'Modules',
            type: 'string',
            width: '10px',
            renderer: columnRenderers.modulesSummary,
            sortFunction: (modules) => sortModules(modules),
            filterFunction: (filterValue) => createModuleFilter(filterValue),
            filterTooltip: 'Filter by module. Separate with , (OR) or & (AND). Use * for wildcards. Prefix @ for all-match, ! to exclude. Examples: *mlp*,*attn* (OR), *mlp*&*attn* (AND), @*proj* (all), !*o_proj* (exclude)'
        }
    ];

    // Add histogram columns dynamically
    const statColors = {
        'all_activations': '#4169E1',
        'max_activation-max-16': '#DC143C',
        'max_activation-max-32': '#DC143C',
        'mean_activation-max-16': '#228B22',
        'median_activation-max-16': '#FF8C00',
        'min_activation-max-16': '#9370DB',
        'max_activation_position': '#FF6347'
    };

    histogramStats.forEach(statKey => {
        const color = statColors[statKey] || '#808080';
        const label = statKey.replace(/-/g, ' ').replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

        columns.push({
            id: 'histogram_' + statKey,
            key: 'stats',
            label: label,
            type: 'number',
            width: '200px',
            align: 'center',
            renderer: columnRenderers.genericHistogram(statKey, color, label),
            sortFunction: (value, row) => {
                const histData = row.stats[statKey];
                if (!histData || !histData.bin_counts || !histData.bin_edges) return -Infinity;
                // For all_activations, use precomputed mean
                if (statKey === 'all_activations' && row.stats.mean_activation !== undefined) {
                    return row.stats.mean_activation;
                }
                // Otherwise calculate mean from histogram
                return calculateHistogramMean(histData);
            },
            filterFunction: (filterValue) => createHistogramFilter(statKey, filterValue),
            filterTooltip: 'Filter by statistics. Use: mean>0.5, median<0.2, max>=1.0, min>-0.1, range<5, sum>100. Combine with commas (e.g., mean>0.5, max<10)'
        });
    });

    // Token activation columns
    columns.push({
        id: 'top_tokens',
        key: 'stats',
        label: 'Top Tokens',
        type: 'string',
        width: '150px',
        align: 'left',
        renderer: columnRenderers.topToken,
        sortFunction: (value, row) => sortTopToken(value, row),
        filterFunction: (filterValue) => createTopTokenFilter(filterValue),
        filterTooltip: 'Search for tokens (case-insensitive substring match)'
    });

    columns.push({
        id: 'token_entropy',
        key: 'stats',
        label: 'Token Entropy',
        type: 'number',
        width: '60px',
        align: 'right',
        renderer: columnRenderers.tokenEntropy,
        sortFunction: (value, row) => {
            const tokenStats = row.stats.token_activations;
            return tokenStats ? tokenStats.entropy : -Infinity;
        },
        filterFunction: (filterValue) => createNumericFilter(filterValue, (stats) => {
            const tokenStats = stats?.token_activations;
            return tokenStats ? tokenStats.entropy : null;
        }),
        filterTooltip: 'Filter by entropy. Use operators: >, <, >=, <=, ==, != (e.g., >2.5)'
    });

    columns.push({
        id: 'token_concentration',
        key: 'stats',
        label: 'Token Conc.',
        type: 'number',
        width: '60px',
        align: 'right',
        renderer: columnRenderers.tokenConcentration,
        sortFunction: (value, row) => {
            const tokenStats = row.stats.token_activations;
            return tokenStats ? tokenStats.concentration_ratio : -Infinity;
        },
        filterFunction: (filterValue) => createNumericFilter(filterValue, (stats) => {
            const tokenStats = stats?.token_activations;
            return tokenStats ? tokenStats.concentration_ratio : null;
        }),
        filterTooltip: 'Filter by concentration (0-1). Use operators: >, <, >=, <=, ==, != (e.g., >0.5)'
    });

    // Explanation column
    columns.push({
        key: 'explanation',
        label: 'Explanation',
        type: 'string',
        width: '200px',
        align: 'left',
        renderer: columnRenderers.explanation,
        filterTooltip: 'Filter by explanation text (case-insensitive substring match)'
    });

    // Actions column
    columns.push({
        key: 'id',
        label: 'Actions',
        type: 'string',
        width: '20px',
        align: 'center',
        renderer: columnRenderers.clusterLink,
        filterable: false
    });

    const tableConfig = {
        data: tableData,
        columns: columns,
        pageSize: CONFIG.indexPage.pageSize,
        pageSizeOptions: CONFIG.indexPage.pageSizeOptions,
        showFilters: CONFIG.indexPage.showFilters
    };

    dataTable = new DataTable('#clusterTableContainer', tableConfig);

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
    } else {
        // Manually trigger Alpine component's loadData now that CONFIG is ready
        const modelInfoEl = document.getElementById('modelInfo');
        if (modelInfoEl && Alpine.$data(modelInfoEl)) {
            Alpine.$data(modelInfoEl).loadData();
        }
    }

    // Load cluster data and render table
    loadData();
});
