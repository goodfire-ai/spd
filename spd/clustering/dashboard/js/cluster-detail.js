let clusterData = null;
let allClusters = null;
let textSamples = {};
let activationsArray = null;
let activationsMap = {};
let currentClusterHash = null;
let modelInfo = {};
let explanations = {};

// Component-level data
let componentActivations = {};  // Map component labels to their activation data
let enabledComponents = new Set();  // Track which components are enabled
let combinationStrategy = 'max';  // How to combine component activations: 'max', 'sum', 'mean'

async function init() {
    // Get cluster hash from URL
    const urlParams = new URLSearchParams(window.location.search);
    currentClusterHash = urlParams.get('id');

    if (!currentClusterHash) {
        document.getElementById('loading').textContent = 'No cluster ID specified';
        return;
    }

    await loadData();
}

async function loadData() {
    const progressBar = NOTIF.pbar('Loading cluster data...');

    try {
        progressBar.progress(0.1);

        // Load data in parallel
        let clusters, samples, activationsMapResponse, modelInfoResponse;

        const clustersPath = CONFIG.getDataPath('clusters');
        const textSamplesPath = CONFIG.getDataPath('textSamples');
        const activationsMapPath = CONFIG.getDataPath('activationsMap');
        const modelInfoPath = CONFIG.getDataPath('modelInfo');
        const explanationsPath = CONFIG.getDataPath('explanations');

        try {
            [clusters, samples, activationsMapResponse, modelInfoResponse] = await Promise.all([
                loadJSONL(clustersPath, 'cluster_hash').catch(e => {
                    throw new Error(`Failed to load ${clustersPath}: ${e.message}`);
                }),
                loadJSONL(textSamplesPath, 'text_hash').catch(e => {
                    throw new Error(`Failed to load ${textSamplesPath}: ${e.message}`);
                }),
                fetch(activationsMapPath).catch(e => {
                    throw new Error(`Failed to load ${activationsMapPath}: ${e.message}`);
                }),
                fetch(modelInfoPath).catch(e => {
                    throw new Error(`Failed to load ${modelInfoPath}: ${e.message}`);
                })
            ]);

            // Load explanations (non-critical, don't fail if missing)
            explanations = await loadJSONL(explanationsPath, 'cluster_id').catch(() => ({}));
        } catch (error) {
            progressBar.complete();
            NOTIF.error(error.message, error, null);
            document.getElementById('loading').textContent = error.message;
            throw error;
        }

        progressBar.progress(0.4);

        if (!activationsMapResponse.ok) {
            const msg = `Failed to load ${activationsMapPath} (HTTP ${activationsMapResponse.status})`;
            NOTIF.error(msg, null, null);
            throw new Error(msg);
        }
        if (!modelInfoResponse.ok) {
            const msg = `Failed to load ${modelInfoPath} (HTTP ${modelInfoResponse.status})`;
            NOTIF.error(msg, null, null);
            throw new Error(msg);
        }

        allClusters = clusters;
        textSamples = samples;

        try {
            activationsMap = await activationsMapResponse.json();
        } catch (error) {
            const msg = `Failed to parse ${activationsMapPath} (invalid JSON)`;
            NOTIF.error(msg, error, null);
            throw new Error(msg);
        }

        try {
            modelInfo = await modelInfoResponse.json();
        } catch (error) {
            const msg = `Failed to parse ${modelInfoPath} (invalid JSON)`;
            NOTIF.error(msg, error, null);
            throw new Error(msg);
        }

        progressBar.progress(0.6);

        if (!allClusters[currentClusterHash]) {
            const msg = 'Cluster not found';
            NOTIF.error(msg, null, null);
            document.getElementById('loading').textContent = msg;
            progressBar.complete();
            return;
        }

        clusterData = allClusters[currentClusterHash];

        // Load activations (float16 compressed npz)
        const activationsPath = CONFIG.getDataPath('activations');
        try {
            activationsArray = await NDArray.load(activationsPath);
        } catch (error) {
            const msg = `Failed to load ${activationsPath}`;
            NOTIF.error(msg, error, null);
            throw new Error(msg);
        }

        progressBar.progress(0.9);

        displayCluster();
        progressBar.complete();
        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        progressBar.complete();
        console.error('Load error:', error);
        console.error('Stack:', error.stack);
    }
}

function displayCluster() {
    // Update title
    document.getElementById('clusterTitle').textContent = `Cluster ${currentClusterHash}`;

    // Display component count
    const componentCount = document.getElementById('componentCount');
    componentCount.textContent = clusterData.components.length;

    // Display explanation and setup copy handler
    displayExplanation();
    setupCopyHandler();

    // Initialize component data
    initializeComponentData();

    // Display model visualization
    displayModelVisualization();

    // Setup components table
    setupComponentsTable();

    // Setup hover highlighting between model view and components table
    setupModelViewHighlighting();

    // Display histogram plots
    displayHistograms();

    // Display token activation stats if available
    if (clusterData.stats && clusterData.stats.token_activations) {
        displayTokenActivations();
    }

    // Display samples
    displaySamples();
}

function displayExplanation() {
    const explanationSpan = document.getElementById('clusterExplanation');
    if (!explanationSpan) return;

    const explanationData = explanations[currentClusterHash];
    if (explanationData && explanationData.explanation) {
        explanationSpan.textContent = explanationData.explanation;
        explanationSpan.style.fontStyle = 'normal';
        explanationSpan.style.color = '#000';
    } else {
        explanationSpan.textContent = 'No explanation';
        explanationSpan.style.fontStyle = 'italic';
        explanationSpan.style.color = '#666';
    }
}

function setupCopyHandler() {
    const copyBtn = document.getElementById('copyTemplateBtn');
    if (!copyBtn) return;

    copyBtn.addEventListener('click', async () => {
        const template = JSON.stringify({
            cluster_id: currentClusterHash,
            explanation: ""
        }) + '\n';

        try {
            await navigator.clipboard.writeText(template);
            NOTIF.success('Template copied to clipboard!');
        } catch (err) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = template;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            document.body.appendChild(textArea);
            textArea.select();
            try {
                document.execCommand('copy');
                NOTIF.success('Template copied to clipboard!');
            } catch (e) {
                NOTIF.error('Failed to copy template', e, null);
            }
            document.body.removeChild(textArea);
        }
    });
}

function initializeComponentData() {
    // Load component activations if available
    if (clusterData.component_activations) {
        componentActivations = clusterData.component_activations;
    }

    // Enable all components by default
    enabledComponents.clear();
    clusterData.components.forEach(comp => {
        enabledComponents.add(comp.label);
    });
}

function displayModelVisualization() {
    const modelViewDiv = document.getElementById('modelView');
    renderModelView(modelViewDiv, currentClusterHash, allClusters, modelInfo, CONFIG.visualization.colormap, CONFIG.visualization.modelViewCellSize);
}

function displayHistograms() {
    const stats = clusterData.stats;
    if (!stats) return;

    const histogramPlots = document.getElementById('histogramPlots');
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
    document.getElementById('tokenActivations').style.display = 'block';

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

function setupComponentsTable() {
    const tableData = clusterData.components.map(comp => ({
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
        displaySamples();
        return;
    }

    // If all components are enabled, use cluster-level activations (faster)
    if (enabledComponents.size === clusterData.components.length) {
        displaySamples();
        return;
    }

    // Recompute activations based on enabled components
    displaySamples();
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
                if (cells.length > 0) {
                    const moduleCell = cells[0]; // First column is module name
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

function displaySamples() {
    const tbody = document.getElementById('samplesTableBody');
    tbody.innerHTML = '';

    // Get the main criterion samples (max_activation)
    const criterionKey = Object.keys(clusterData.criterion_samples)[0];
    if (!criterionKey) {
        tbody.innerHTML = '<tr><td colspan="2">No samples available</td></tr>';
        return;
    }

    const sampleHashes = clusterData.criterion_samples[criterionKey];
    const samplesToShow = Math.min(CONFIG.clusterPage.maxSamplesPerCluster, sampleHashes.length);

    // Check if we need to use component-level activations
    const useComponentActivations = componentActivations &&
                                     Object.keys(componentActivations).length > 0 &&
                                     enabledComponents.size < clusterData.components.length;

    for (let i = 0; i < samplesToShow; i++) {
        const textHash = sampleHashes[i];
        const textSample = textSamples[textHash];

        if (!textSample) {
            console.warn(`Text sample not found for hash: ${textHash}`);
            continue;
        }

        let activationsData;

        if (useComponentActivations) {
            // Compute combined activations from enabled components
            const componentActsList = [];

            for (const comp of clusterData.components) {
                if (enabledComponents.has(comp.label) && componentActivations[comp.label]) {
                    const compData = componentActivations[comp.label];
                    // Find the activation for this text sample
                    const hashIdx = compData.activation_sample_hashes.indexOf(`${currentClusterHash}:${comp.label}:${textHash}`);
                    if (hashIdx !== -1) {
                        const activationIdx = compData.activation_indices[hashIdx];
                        if (activationIdx !== undefined && activationsArray) {
                            const compActivations = activationsArray.get(activationIdx);
                            componentActsList.push(Array.from(compActivations.data));
                        }
                    }
                }
            }

            if (componentActsList.length > 0) {
                activationsData = combineComponentActivations(componentActsList, combinationStrategy);
            }
        }

        // Fall back to cluster-level activations if component activations not available
        if (!activationsData) {
            const fullHash = `${currentClusterHash}:${textHash}`;
            const activationIdx = activationsMap[fullHash];

            if (activationIdx !== undefined && activationsArray) {
                const activations = activationsArray.get(activationIdx);
                activationsData = Array.from(activations.data);
            }
        }

        let tokenViz;
        if (activationsData) {
            // Find max position
            const maxPosition = activationsData.indexOf(Math.max(...activationsData));

            // Use the proper token visualization with coloring and tooltips
            tokenViz = createTokenVisualizationWithTooltip(
                textSample.tokens,
                activationsData,
                maxPosition
            );
        } else {
            // Fallback to simple visualization if no activations
            console.warn(`No activations found for sample ${i}`);
            tokenViz = createSimpleTokenViz(textSample.tokens);
        }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td></td>
        `;

        // Add token visualization to last cell
        tr.lastElementChild.appendChild(tokenViz);

        tbody.appendChild(tr);
    }

    if (sampleHashes.length > CONFIG.clusterPage.maxSamplesPerCluster) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="2" style="text-align: center;">
            ... and ${sampleHashes.length - CONFIG.clusterPage.maxSamplesPerCluster} more samples
        </td>`;
        tbody.appendChild(tr);
    }
}

function createSimpleTokenViz(tokens) {
    const container = document.createElement('div');
    container.className = 'token-container';
    container.textContent = tokens.join(' ');
    return container;
}

// Initialize config and load data on page load
(async () => {
    await initConfig();
    init();
})();