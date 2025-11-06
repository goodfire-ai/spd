let componentData = null;
let allComponents = [];
let currentComponentLabel = null;
let dashboardData = {};
let fullData = null;  // Store full ZANJ data for accessing activations
// TODO: Re-enable explanations feature
// let explanations = {};

async function init() {
    // Get component label from URL
    const urlParams = new URLSearchParams(window.location.search);
    currentComponentLabel = urlParams.get('label');

    if (!currentComponentLabel) {
        const loading = document.getElementById('loading');
        if (!loading) {
            const msg = 'Fatal error: loading element not found in HTML';
            NOTIF.error(msg, null, null);
            console.error(msg);
            return;
        }
        loading.textContent = 'No component label specified';
        return;
    }

    await loadData();
}

async function loadData() {
    try {
        // Load all data via ZANJ
        const loader = new ZanjLoader(CONFIG.data.dataDir);
        const data = await loader.read();
        fullData = data;  // Store for accessing activations later

        dashboardData = await data.metadata;

        // Extract module name from component label (format: "module.name:index")
        const moduleName = currentComponentLabel.split(':')[0];

        // Load components from index_summaries and filter by module
        const indexSummaries = await data.index_summaries;
        const allSummaries = await indexSummaries.summaries;
        allComponents = allSummaries.filter(comp => comp.label.split(':')[0] === moduleName);

        // TODO: Re-enable explanations feature
        // Load explanations separately (not part of ZANJ)
        // const explanationsPath = CONFIG.getDataPath('explanations');
        // explanations = await loadJSONL(explanationsPath, 'component_label').catch(() => ({}));

        // Find the component with matching label
        componentData = allComponents.find(comp => comp.label === currentComponentLabel);

        if (!componentData) {
            const msg = 'Component not found';
            NOTIF.error(msg, null, null);
            const loading = document.getElementById('loading');
            if (loading) {
                loading.textContent = msg;
            } else {
                console.error('loading element not found, cannot display error message');
            }
            return;
        }

        displayComponent();
        const loading = document.getElementById('loading');
        if (!loading) {
            const msg = 'Fatal error: loading element not found in HTML';
            NOTIF.error(msg, null, null);
            console.error(msg);
            return;
        }
        loading.style.display = 'none';
    } catch (error) {
        console.error('Load error:', error);
        console.error('Stack:', error.stack);
        NOTIF.error('Failed to load component data: ' + error.message, error, null);
    }
}

async function displayComponent() {
    // Update title
    const componentTitle = document.getElementById('componentTitle');
    if (!componentTitle) {
        const msg = 'Fatal error: componentTitle element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    componentTitle.textContent = `Component: ${currentComponentLabel}`;

    // Display module and index
    const componentModule = document.getElementById('componentModule');
    const componentIndex = document.getElementById('componentIndex');
    if (componentModule && componentIndex) {
        const labelParts = currentComponentLabel.split(':');
        componentModule.textContent = labelParts[0];
        componentIndex.textContent = labelParts.length > 1 ? labelParts[1] : '0';
    }

    // TODO: Re-enable explanations feature
    // Display explanation and setup copy handler
    // displayExplanation();
    // setupCopyHandler();

    // Display model visualization (single component)
    await displayModelVisualization();

    // Display histogram plots
    displayHistograms();

    // Display token activation stats if available (when backend adds it)
    if (componentData.stats && componentData.stats.token_activations) {
        displayTokenActivations();
    }

    // Display token statistics
    await displayTokenStatistics();

    // Display top samples
    await displaySamples();
}

async function displayModelVisualization() {
    const modelViewDiv = document.getElementById('modelView');
    if (!modelViewDiv) {
        const msg = 'Fatal error: modelView element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    // For a single component, we might want to just show where it is in the model
    // For now, let's just display a simple text representation
    modelViewDiv.innerHTML = `<p>Component location: ${currentComponentLabel}</p>`;
    // TODO: Implement single-component model visualization
}

function displayHistograms() {
    const histograms = componentData.histograms;
    if (!histograms) return;

    const histogramPlots = document.getElementById('histogramPlots');
    if (!histogramPlots) {
        const msg = 'Fatal error: histogramPlots element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    histogramPlots.innerHTML = '';

    // Color mapping for different histogram types
    const statColors = {
        'all_activations': '#4169E1',
        'max_per_sample': '#DC143C',
        'mean_per_sample': '#228B22'
    };

    // Create a plot for each histogram
    Object.entries(histograms).forEach(([histKey, histData]) => {
        const color = statColors[histKey] || '#808080';
        const label = histKey.replace(/-/g, ' ').replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

        // Create container for this plot
        const plotContainer = document.createElement('div');
        plotContainer.style.display = 'flex';
        plotContainer.style.flexDirection = 'column';
        plotContainer.style.alignItems = 'center';
        plotContainer.style.minWidth = '250px';

        // Add label
        const plotLabel = document.createElement('div');
        plotLabel.textContent = label;
        plotLabel.style.fontSize = '12px';
        plotLabel.style.fontWeight = 'bold';
        plotLabel.style.marginBottom = '5px';
        plotLabel.style.textAlign = 'center';
        plotContainer.appendChild(plotLabel);

        // Create sparkline
        const sparklineContainer = document.createElement('div');
        sparklineContainer.className = 'sparkline-cell';

        // Calculate bin centers for x-axis
        const binCenters = calculateBinCenters(histData.edges);

        // Use configured range when data fits, otherwise use actual edge range
        // Use epsilon tolerance for floating point comparison
        const min = histData.edges[0];
        const max = histData.edges[histData.edges.length - 1];
        const configRange = CONFIG.visualization.histogramRange;
        const epsilon = 1e-6;
        const xlims = configRange &&
                      (min >= configRange[0] - epsilon && max <= configRange[1] + epsilon)
            ? configRange
            : [min, max];

        const svg = sparkbars(binCenters, histData.counts, {
            width: CONFIG.visualization.sparklineWidth || 200,
            height: CONFIG.visualization.sparklineHeight || 60,
            color: color,
            shading: true,
            lineWidth: 0,
            markers: '',
            margin: 2,
            xlims: xlims,
            ylims: [0, null],
            logScale: true,
            xAxis: {line: true, ticks: true, label_margin: 10},
            yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin || 35}
        });

        sparklineContainer.innerHTML = svg;

        // Add tooltip with statistics
        const mean = calculateHistogramMean(histData);
        const median = calculateHistogramMedian(histData);
        const totalCount = histData.counts.reduce((a, b) => a + b, 0);
        sparklineContainer.title = `${label} (n=${totalCount})\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

        plotContainer.appendChild(sparklineContainer);
        histogramPlots.appendChild(plotContainer);
    });
}

function displayTokenActivations() {
    const tokenStats = componentData.stats.token_activations;

    // Show the section
    const tokenActivations = document.getElementById('tokenActivations');
    if (!tokenActivations) {
        const msg = 'Fatal error: tokenActivations element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    tokenActivations.style.display = 'block';

    // Setup top tokens table
    if (tokenStats.top_tokens && tokenStats.top_tokens.length > 0) {
        const tableData = tokenStats.top_tokens.map((item, idx) => ({
            rank: idx + 1,
            token: item.token,
            count: item.count,
            percentage: ((item.count / tokenStats.total_activations) * 100)
        }));

        const maxPercentage = tableData.length > 0 ? tableData[0].percentage : 0;

        const tableConfig = {
            data: tableData,
            columns: [
                {
                    key: 'rank',
                    label: '#',
                    type: 'number',
                    width: '40px',
                    align: 'right'
                },
                {
                    key: 'token',
                    label: 'Token',
                    type: 'string',
                    width: '120px',
                    renderer: (value) => {
                        // Show token in a monospace box with visual formatting
                        const tokenDisplay = value.replace(/ /g, '·').replace(/\n/g, '↵');
                        return `<code class="token-display">${tokenDisplay}</code>`;
                    }
                },
                {
                    key: 'percentage',
                    label: '%',
                    type: 'number',
                    width: '70px',
                    align: 'right',
                    renderer: (value) => {
                        const percentageValue = value;
                        const percentage = percentageValue.toFixed(1);

                        // Color based on percentage (normalized by max percentage)
                        const normalizedPct = maxPercentage > 0 ? percentageValue / maxPercentage : 0;
                        const intensity = Math.floor((1 - normalizedPct) * 255);
                        const bgColor = `rgb(255, ${intensity}, ${intensity})`;

                        const span = document.createElement('span');
                        span.textContent = `${percentage}%`;
                        span.style.backgroundColor = bgColor;
                        span.style.padding = '2px 4px';
                        span.style.borderRadius = '2px';

                        return span;
                    },
                    infoFunction: () => {
                        return `Unique: ${tokenStats.total_unique_tokens.toLocaleString()} | Total: ${tokenStats.total_activations.toLocaleString()} | Entropy: ${tokenStats.entropy.toFixed(2)} | Conc: ${(tokenStats.concentration_ratio * 100).toFixed(1)}%`;
                    }
                }
            ],
            pageSize: 10,
            showFilters: false,
            showInfo: true
        };

        new DataTable('#topTokensTable', tableConfig);
    }
}

async function displayTokenStatistics() {
    // Try accessing synchronously first, then await if needed
    let tokenStats;
    try {
        tokenStats = componentData.token_stats;
        // Access a property to trigger the error if not loaded
        const _check = tokenStats.length;
    } catch (error) {
        if (error.message && error.message.includes('use await')) {
            tokenStats = await componentData.token_stats;
        } else {
            throw error;
        }
    }

    if (!tokenStats || tokenStats.length === 0) {
        return;
    }

    // Validate structure
    const firstToken = tokenStats[0];
    console.assert(
        firstToken.token !== undefined &&
        firstToken.p_token_given_active !== undefined &&
        firstToken.p_active_given_token !== undefined &&
        firstToken.count_when_active !== undefined &&
        firstToken.count_token_total !== undefined,
        'Token stat structure is invalid'
    );

    // Create table with both probability columns
    const tableConfig = {
        data: tokenStats,
        columns: [
            {
                key: 'token',
                label: 'Token',
                type: 'string',
                width: '120px',
                renderer: (value) => `<code class="token-display">${formatTokenDisplay(value)}</code>`
            },
            {
                key: 'p_token_given_active',
                label: 'P(token|active)',
                type: 'number',
                width: '120px',
                align: 'right',
                renderer: (value) => (value * 100).toFixed(2) + '%'
            },
            {
                key: 'p_active_given_token',
                label: 'P(active|token)',
                type: 'number',
                width: '120px',
                align: 'right',
                renderer: (value) => (value * 100).toFixed(2) + '%'
            },
            {
                key: 'count_when_active',
                label: 'Count (active)',
                type: 'number',
                width: '100px',
                align: 'right'
            },
            {
                key: 'count_token_total',
                label: 'Count (total)',
                type: 'number',
                width: '100px',
                align: 'right'
            }
        ],
        pageSize: 20,
        showFilters: true,
        showInfo: true
    };

    new DataTable('#tokenStatsTable', tableConfig);
}

async function displaySamples() {
    const container = document.getElementById('topSamplesTable');
    if (!container) return;

    try {
        // Load samples using utility function
        const samples = await loadComponentSamples(fullData, currentComponentLabel, 1000);

        if (samples.length === 0) {
            container.innerHTML = '<p>No samples available</p>';
            return;
        }

        console.log(`Loaded ${samples.length} samples`);

        // Add sequence index and compute stats
        const allSamples = samples.map((sample, seq) => {
            const maxAct = Math.max(...sample.activations);
            const meanAct = sample.activations.reduce((a, b) => a + b, 0) / sample.activations.length;
            return {
                sequence_index: seq,
                max_act: maxAct,
                mean_act: meanAct,
                token_strs: sample.token_strs,
                activations: sample.activations
            };
        });

        // Create sortable DataTable
        new DataTable(container, {
            data: allSamples,
            columns: [
                { key: 'max_act', label: 'Max Act', type: 'number', width: '100px',
                  renderer: (val) => val.toFixed(4) },
                { key: 'mean_act', label: 'Mean Act', type: 'number', width: '100px',
                  renderer: (val) => val.toFixed(4) },
                { key: 'token_strs', label: 'Text', type: 'string', filterable: false,
                  renderer: (val, row) => createTokenVisualization(row.token_strs, row.activations) }
            ],
            pageSize: 20,
            showFilters: true,
            defaultSort: { key: 'max_act', direction: 'desc' }
        });

        // Add token hover highlighting after table is rendered
        setupTokenHighlighting();
    } catch (error) {
        console.error('Error in displaySamples:', error);
        console.error('Stack:', error.stack);
        container.innerHTML = '<p>Error loading samples: ' + error.message + '</p>';
    }
}

/**
 * Setup hover event listeners for token cross-highlighting
 * When hovering over a token, all instances of that token are highlighted
 */
function setupTokenHighlighting() {
    // Find all token elements across all tables
    const allTokens = document.querySelectorAll('.token');

    allTokens.forEach(tokenEl => {
        tokenEl.addEventListener('mouseenter', function() {
            const tokenText = this.getAttribute('data-token');
            // Find and highlight all tokens with matching text
            const matchingTokens = document.querySelectorAll(`.token[data-token="${CSS.escape(tokenText)}"]`);
            matchingTokens.forEach(match => match.classList.add('token-highlighted'));
        });

        tokenEl.addEventListener('mouseleave', function() {
            // Remove highlight from all tokens
            document.querySelectorAll('.token-highlighted').forEach(el => {
                el.classList.remove('token-highlighted');
            });
        });
    });
}


// Initialize config and load data on page load
document.addEventListener('DOMContentLoaded', async () => {
    await initConfig();
    init();
});