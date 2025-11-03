let componentData = null;
let allComponents = [];
let currentComponentLabel = null;
let dashboardData = {};
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

        dashboardData = await data.metadata;

        // Extract module name from component label (format: "module.name:index")
        const moduleName = currentComponentLabel.split(':')[0];

        // Load only the specific module's components (lazy loading)
        const subcomponentDetails = await data.subcomponent_details;
        allComponents = await subcomponentDetails[moduleName];

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
    displayTokenStatistics();

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

        // Enforce [0, 1] bounds for clean x-axis labels (backend handles binning via config.hist_range)
        const svg = sparkbars(binCenters, histData.counts, {
            width: CONFIG.visualization.sparklineWidth || 200,
            height: CONFIG.visualization.sparklineHeight || 60,
            color: color,
            shading: true,
            lineWidth: 0,
            markers: '',
            margin: 2,
            xlims: [0, 1],
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

function displayTokenStatistics() {
    const topN = CONFIG.tokenStats.detailTopN;

    // Helper to create table config
    const createTableConfig = (data, probabilityKey, probabilityLabel) => ({
        data: data.slice(0, topN).map((item, idx) => ({
            rank: idx + 1,
            token: item.token,
            probability: item[probabilityKey],
            count_when_active: item.count_when_active,
            count_token_total: item.count_token_total
        })),
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
                renderer: (value) => `<code class="token-display">${formatTokenDisplay(value)}</code>`
            },
            {
                key: 'probability',
                label: probabilityLabel,
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
        pageSize: 10,
        showFilters: false,
        showInfo: true
    });

    // Display P(token | active) table
    if (componentData.top_tokens_given_active && componentData.top_tokens_given_active.length > 0) {
        // Validate structure
        const firstToken = componentData.top_tokens_given_active[0];
        console.assert(
            firstToken.token !== undefined &&
            firstToken.p_token_given_active !== undefined &&
            firstToken.count_when_active !== undefined &&
            firstToken.count_token_total !== undefined,
            'Token stat structure is invalid for top_tokens_given_active'
        );

        const tableConfig = createTableConfig(
            componentData.top_tokens_given_active,
            'p_token_given_active',
            'P(token|active)'
        );
        new DataTable('#topTokensGivenActiveTable', tableConfig);
    }

    // Display P(active | token) table
    if (componentData.top_active_given_tokens && componentData.top_active_given_tokens.length > 0) {
        const tableConfig = createTableConfig(
            componentData.top_active_given_tokens,
            'p_active_given_token',
            'P(active|token)'
        );
        new DataTable('#topActiveGivenTokensTable', tableConfig);
    }
}

async function displaySamples() {
    const container = document.getElementById('topSamplesTable');
    if (!container) return;

    // Iterate over all sample types in the top_samples dict
    const allSamples = [];

    for (const [sampleType, samples] of Object.entries(componentData.top_samples)) {
        for (let i = 0; i < samples.length; i++) {
            const sample = samples[i];
            const activations = await sample.activations;
            const activationsArray = activations.data
                ? Array.from(activations.data)
                : (Array.isArray(activations) ? activations : Array.from(activations));

            allSamples.push({
                type: sampleType.replace('top_', ''),  // "top_max" -> "max"
                rank: i + 1,
                max_act: Math.max(...activationsArray),
                mean_act: activationsArray.reduce((a, b) => a + b, 0) / activationsArray.length,
                token_strs: sample.token_strs,
                activations: activationsArray
            });
        }
    }

    // Create DataTable with custom renderer for text column
    new DataTable(container, {
        data: allSamples,
        columns: [
            { key: 'type', label: 'Type', type: 'string', width: '80px' },
            { key: 'rank', label: 'Rank', type: 'number', width: '80px' },
            { key: 'max_act', label: 'Max Act', type: 'number', width: '100px',
              renderer: (val) => val.toFixed(4) },
            { key: 'mean_act', label: 'Mean Act', type: 'number', width: '100px',
              renderer: (val) => val.toFixed(4) },
            { key: 'token_strs', label: 'Text', type: 'string', filterable: false,
              renderer: (val, row) => {
                  return createTokenVisualization(row.token_strs, row.activations);
              }
            }
        ],
        pageSize: 10,
        showFilters: true
    });

    // Add token hover highlighting after table is rendered
    setupTokenHighlighting();
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