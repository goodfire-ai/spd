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

        dashboardData = data;
        allComponents = await data.components;

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

        const min = histData.edges[0];
        const max = histData.edges[histData.edges.length - 1];

        // Set x-axis limits to [0, 1] if data is in that range
        const xlims = (min >= 0 && max <= 1) ? [0, 1] : null;

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

async function displaySamples() {
    // Display top_max samples
    const topMaxBody = document.getElementById('topMaxTableBody');
    if (topMaxBody) {
        topMaxBody.innerHTML = '';
        const topMaxSamples = componentData.top_max || [];

        if (topMaxSamples.length === 0) {
            topMaxBody.innerHTML = '<tr><td colspan="2">No samples available</td></tr>';
        } else {
            for (let i = 0; i < topMaxSamples.length; i++) {
                const sample = topMaxSamples[i];
                const tokenViz = createTokenVisualization(
                    sample.token_strs,
                    Array.from(sample.activations)
                );

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${i + 1}</td>
                    <td></td>
                `;

                // Add token visualization to last cell
                tr.lastElementChild.appendChild(tokenViz);
                topMaxBody.appendChild(tr);
            }
        }
    }

    // Display top_mean samples
    const topMeanBody = document.getElementById('topMeanTableBody');
    if (topMeanBody) {
        topMeanBody.innerHTML = '';
        const topMeanSamples = componentData.top_mean || [];

        if (topMeanSamples.length === 0) {
            topMeanBody.innerHTML = '<tr><td colspan="2">No samples available</td></tr>';
        } else {
            for (let i = 0; i < topMeanSamples.length; i++) {
                const sample = topMeanSamples[i];
                const tokenViz = createTokenVisualization(
                    sample.token_strs,
                    Array.from(sample.activations)
                );

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${i + 1}</td>
                    <td></td>
                `;

                // Add token visualization to last cell
                tr.lastElementChild.appendChild(tokenViz);
                topMeanBody.appendChild(tr);
            }
        }
    }
}


// Initialize config and load data on page load
document.addEventListener('DOMContentLoaded', async () => {
    await initConfig();
    init();
});