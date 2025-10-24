let clusterData = null;
let allClusters = null;
let textSamples = {};
let currentClusterHash = null;
let modelInfo = {};
// TODO: Re-enable explanations feature
// let explanations = {};

// DEPRECATED: activationsArray and activationsMap no longer needed - data is self-contained in cluster.samples

// Component-level data
let componentActivations = {};  // Map component labels to their activation data
let enabledComponents = new Set();  // Track which components are enabled
let combinationStrategy = 'max';  // How to combine component activations: 'max', 'sum', 'mean'

async function init() {
    // Get cluster hash from URL
    const urlParams = new URLSearchParams(window.location.search);
    currentClusterHash = urlParams.get('id');

    if (!currentClusterHash) {
        const loading = document.getElementById('loading');
        if (!loading) {
            const msg = 'Fatal error: loading element not found in HTML';
            NOTIF.error(msg, null, null);
            console.error(msg);
            return;
        }
        loading.textContent = 'No cluster ID specified';
        return;
    }

    await loadData();
}

async function loadData() {
    try {
        // Load all data via ZANJ
        const loader = new ZanjLoader(CONFIG.data.dataDir);
        const data = await loader.read();

        allClusters = data.clusters;
        textSamples = data.text_samples;
        modelInfo = data.model_info;

        // TODO: Re-enable explanations feature
        // Load explanations separately (not part of ZANJ)
        // const explanationsPath = CONFIG.getDataPath('explanations');
        // explanations = await loadJSONL(explanationsPath, 'cluster_id').catch(() => ({}));

        if (!allClusters[currentClusterHash]) {
            const msg = 'Cluster not found';
            NOTIF.error(msg, null, null);
            const loading = document.getElementById('loading');
            if (loading) {
                loading.textContent = msg;
            } else {
                console.error('loading element not found, cannot display error message');
            }
            return;
        }

        clusterData = allClusters[currentClusterHash];

        displayCluster();
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
        NOTIF.error('Failed to load cluster data: ' + error.message, error, null);
    }
}

async function displayCluster() {
    // Update title
    const clusterTitle = document.getElementById('clusterTitle');
    if (!clusterTitle) {
        const msg = 'Fatal error: clusterTitle element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    clusterTitle.textContent = `Cluster ${currentClusterHash}`;

    // Display component count
    const componentCount = document.getElementById('componentCount');
    if (!componentCount) {
        const msg = 'Fatal error: componentCount element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    // Await lazy-loaded components
    const components = await clusterData.components;
    componentCount.textContent = components.length;

    // TODO: Re-enable explanations feature
    // Display explanation and setup copy handler
    // displayExplanation();
    // setupCopyHandler();

    // Initialize component data
    await initializeComponentData();

    // Display model visualization
    displayModelVisualization();

    // Setup components table
    await setupComponentsTable();

    // Setup hover highlighting between model view and components table
    setupModelViewHighlighting();

    // Display histogram plots
    displayHistograms();

    // Display token activation stats if available
    if (clusterData.stats && clusterData.stats.token_activations) {
        displayTokenActivations();
    }

    // Display samples
    await displaySamples();
}

// TODO: Re-enable explanations feature
// function displayExplanation() {
//     const explanationSpan = document.getElementById('clusterExplanation');
//     if (!explanationSpan) return;
//
//     const explanationData = explanations[currentClusterHash];
//     if (explanationData && explanationData.explanation) {
//         explanationSpan.textContent = explanationData.explanation;
//         explanationSpan.style.fontStyle = 'normal';
//         explanationSpan.style.color = '#000';
//     } else {
//         explanationSpan.textContent = 'No explanation';
//         explanationSpan.style.fontStyle = 'italic';
//         explanationSpan.style.color = '#666';
//     }
// }
//
// function setupCopyHandler() {
//     const copyBtn = document.getElementById('copyTemplateBtn');
//     if (!copyBtn) return;
//
//     copyBtn.addEventListener('click', async () => {
//         const template = JSON.stringify({
//             cluster_id: currentClusterHash,
//             explanation: ""
//         }) + '\n';
//
//         try {
//             await navigator.clipboard.writeText(template);
//             NOTIF.success('Template copied to clipboard!');
//         } catch (err) {
//             // Fallback for older browsers
//             const textArea = document.createElement('textarea');
//             textArea.value = template;
//             textArea.style.position = 'fixed';
//             textArea.style.left = '-999999px';
//             document.body.appendChild(textArea);
//             textArea.select();
//             try {
//                 document.execCommand('copy');
//                 NOTIF.success('Template copied to clipboard!');
//             } catch (e) {
//                 NOTIF.error('Failed to copy template', e, null);
//             }
//             document.body.removeChild(textArea);
//         }
//     });
// }

async function initializeComponentData() {
    // Load component activations if available
    if (clusterData.component_activations) {
        componentActivations = clusterData.component_activations;
    }

    // Enable all components by default
    enabledComponents.clear();
    // Await lazy-loaded components
    const components = await clusterData.components;
    components.forEach(comp => {
        enabledComponents.add(comp.label);
    });
}

function displayModelVisualization() {
    const modelViewDiv = document.getElementById('modelView');
    if (!modelViewDiv) {
        const msg = 'Fatal error: modelView element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    renderModelView(modelViewDiv, currentClusterHash, allClusters, modelInfo, CONFIG.visualization.colormap, CONFIG.visualization.modelViewCellSize);
}

function displayHistograms() {
    const stats = clusterData.stats;
    if (!stats) return;

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
        'max_activation-max-16': '#DC143C',
        'max_activation-max-32': '#DC143C',
        'mean_activation-max-16': '#228B22',
        'median_activation-max-16': '#FF8C00',
        'min_activation-max-16': '#9370DB',
        'max_activation_position': '#FF6347'
    };

    // Discover all histogram stats
    const histogramStats = [];
    for (const [key, value] of Object.entries(stats)) {
        if (value && typeof value === 'object' && 'bin_counts' in value && 'bin_edges' in value) {
            histogramStats.push(key);
        }
    }

    // Create a plot for each histogram stat
    histogramStats.forEach(statKey => {
        const histData = stats[statKey];
        const color = statColors[statKey] || '#808080';
        const label = statKey.replace(/-/g, ' ').replace(/_/g, ' ')
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
        const binCenters = calculateBinCenters(histData.bin_edges);

        const min = histData.bin_edges[0];
        const max = histData.bin_edges[histData.bin_edges.length - 1];

        // Set x-axis limits to [0, 1] if data is in that range
        const xlims = (min >= 0 && max <= 1) ? [0, 1] : null;

        const svg = sparkbars(binCenters, histData.bin_counts, {
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
        const totalCount = histData.bin_counts.reduce((a, b) => a + b, 0);
        sparklineContainer.title = `${label} (n=${totalCount})\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMedian: ${median.toFixed(4)}`;

        plotContainer.appendChild(sparklineContainer);
        histogramPlots.appendChild(plotContainer);
    });
}

function displayTokenActivations() {
    const tokenStats = clusterData.stats.token_activations;

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

async function setupComponentsTable() {
    // Await lazy-loaded components
    const components = await clusterData.components;
    const tableData = components.map(comp => ({
        label: comp.label,
        module: comp.module,
        index: comp.index,
        enabled: enabledComponents.has(comp.label)
    }));

    const tableConfig = {
        data: tableData,
        columns: [
            {
                key: 'enabled',
                label: '✓',
                type: 'boolean',
                width: '40px',
                align: 'center',
                renderer: (value, row) => {
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.checked = value;
                    checkbox.style.cursor = 'pointer';
                    checkbox.addEventListener('change', (e) => {
                        onComponentToggle(row.label, e.target.checked);
                    });
                    return checkbox;
                },
                filterable: false
            },
            {
                key: 'module',
                label: 'Module',
                type: 'string',
                width: '250px'
            },
            {
                key: 'index',
                label: 'Index',
                type: 'number',
                width: '80px',
                align: 'right'
            }
        ],
        pageSize: CONFIG.clusterPage.pageSize,
        showFilters: false
    };

    new DataTable('#componentsTable', tableConfig);
}

function onComponentToggle(componentLabel, isEnabled) {
    if (isEnabled) {
        enabledComponents.add(componentLabel);
    } else {
        enabledComponents.delete(componentLabel);
    }

    // Recompute and redisplay activations
    recomputeDisplayedActivations();
}

async function recomputeDisplayedActivations() {
    // If no components are enabled or component activations not available, use cluster-level
    if (enabledComponents.size === 0 || !componentActivations || Object.keys(componentActivations).length === 0) {
        // Just redisplay with cluster-level activations (default)
        await displaySamples();
        return;
    }

    // Await lazy-loaded components
    const components = await clusterData.components;

    // If all components are enabled, use cluster-level activations (faster)
    if (enabledComponents.size === components.length) {
        await displaySamples();
        return;
    }

    // Recompute activations based on enabled components
    await displaySamples();
}

function combineComponentActivations(componentActsList, strategy) {
    // componentActsList: array of activation arrays [n_ctx]
    // Returns: combined activation array [n_ctx]

    if (componentActsList.length === 0) {
        return null;
    }

    if (componentActsList.length === 1) {
        return componentActsList[0];
    }

    const n_ctx = componentActsList[0].length;
    const combined = new Array(n_ctx).fill(0);

    if (strategy === 'max') {
        for (let i = 0; i < n_ctx; i++) {
            let maxVal = componentActsList[0][i];
            for (let j = 1; j < componentActsList.length; j++) {
                if (componentActsList[j][i] > maxVal) {
                    maxVal = componentActsList[j][i];
                }
            }
            combined[i] = maxVal;
        }
    } else if (strategy === 'sum') {
        for (let i = 0; i < n_ctx; i++) {
            let sum = 0;
            for (let j = 0; j < componentActsList.length; j++) {
                sum += componentActsList[j][i];
            }
            combined[i] = sum;
        }
    } else if (strategy === 'mean') {
        for (let i = 0; i < n_ctx; i++) {
            let sum = 0;
            for (let j = 0; j < componentActsList.length; j++) {
                sum += componentActsList[j][i];
            }
            combined[i] = sum / componentActsList.length;
        }
    }

    return combined;
}

function setupModelViewHighlighting() {
    // Get all model view cells
    const modelViewCells = document.querySelectorAll('.modelview-module-cell');

    // Get components table
    const componentsTable = document.querySelector('#componentsTable');
    if (!componentsTable) return;

    modelViewCells.forEach(cell => {
        cell.addEventListener('mouseenter', (e) => {
            const moduleName = e.target.dataset.module;
            if (!moduleName) return;

            // Find and highlight all rows in the components table that match this module
            const tableRows = componentsTable.querySelectorAll('.tablejs-data-row');
            tableRows.forEach(row => {
                const cells = row.querySelectorAll('td');
                if (cells.length > 1) {
                    const moduleCell = cells[1]; // Second column is module name (first is checkbox)
                    if (moduleCell && moduleCell.textContent === moduleName) {
                        row.style.backgroundColor = '#fff3cd'; // Light yellow highlight
                    }
                }
            });
        });

        cell.addEventListener('mouseleave', () => {
            // Remove highlighting from all rows
            const tableRows = componentsTable.querySelectorAll('.tablejs-data-row');
            tableRows.forEach(row => {
                row.style.backgroundColor = '';
            });
        });
    });
}

async function displaySamples() {
    const tbody = document.getElementById('samplesTableBody');
    if (!tbody) {
        const msg = 'Fatal error: samplesTableBody element not found in HTML';
        NOTIF.error(msg, null, null);
        console.error(msg);
        return;
    }
    tbody.innerHTML = '';

    // Use self-contained samples from cluster data
    const samples = clusterData.samples || [];
    if (samples.length === 0) {
        tbody.innerHTML = '<tr><td colspan="2">No samples available</td></tr>';
        return;
    }

    const samplesToShow = Math.min(CONFIG.clusterPage.maxSamplesPerCluster, samples.length);

    for (let i = 0; i < samplesToShow; i++) {
        const sample = samples[i];
        const textSample = textSamples[sample.text_hash];

        if (!textSample) {
            console.warn(`Text sample not found for hash: ${sample.text_hash}`);
            continue;
        }

        // Debug: Log the actual sample structure
        console.log(`Sample ${i} structure:`, {
            keys: Object.keys(sample),
            sample: sample,
            hasActivations: 'activations' in sample,
            activationsType: typeof sample.activations,
            activationsValue: sample.activations
        });

        // Activations might be a ZANJ Proxy (lazy-loaded .npy reference)
        // Need to await it to get the actual NDArray object
        const activations = await sample.activations;

        // The NDArray object has the actual data in the .data property (Float32Array)
        // Convert to regular array for visualization
        const activationsData = activations.data
            ? Array.from(activations.data)
            : (Array.isArray(activations) ? activations : Array.from(activations));

        // Fail immediately if activations are missing or empty
        if (!activationsData || activationsData.length === 0) {
            console.error('sample:', sample);
            console.error('activations:', activations);
            console.error('activationsData:', activationsData);
            throw new Error(
                `No activations found for sample ${i} in cluster ${currentClusterHash}.\n` +
                `Sample structure: ${JSON.stringify(Object.keys(sample))}\n` +
                `sample.activations type: ${typeof sample.activations}\n` +
                `activations after await type: ${typeof activations}\n` +
                `activationsData length: ${activationsData?.length}\n` +
                `Expected: Array or ArrayLike with length > 0`
            );
        }

        const tokenViz = createTokenVisualization(
            sample.tokens,
            activationsData
        );

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td></td>
        `;

        // Add token visualization to last cell
        tr.lastElementChild.appendChild(tokenViz);

        tbody.appendChild(tr);
    }

    if (samples.length > CONFIG.clusterPage.maxSamplesPerCluster) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="2" style="text-align: center;">
            ... and ${samples.length - CONFIG.clusterPage.maxSamplesPerCluster} more samples
        </td>`;
        tbody.appendChild(tr);
    }
}


// Initialize config and load data on page load
document.addEventListener('DOMContentLoaded', async () => {
    await initConfig();
    init();
});