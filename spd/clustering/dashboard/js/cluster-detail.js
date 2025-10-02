let clusterData = null;
let allClusters = null;
let textSamples = {};
let activationsArray = null;
let activationsMap = {};
let currentClusterHash = null;
let modelInfo = {};

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

        const svg = sparkbars(histData.bin_counts, null, {
            width: CONFIG.visualization.sparklineWidth || 200,
            height: CONFIG.visualization.sparklineHeight || 60,
            color: color,
            shading: true,
            lineWidth: 0,
            markers: '',
            margin: 2,
            ylims: [0, null],
            logScale: true,
            xAxis: {line: true, ticks: true, label_margin: 10},
            yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin || 35}
        });

        sparklineContainer.innerHTML = svg;

        // Add tooltip
        const maxBinCount = Math.max(...histData.bin_counts);
        const min = histData.bin_edges[0];
        const max = histData.bin_edges[histData.bin_edges.length - 1];
        sparklineContainer.title = `${label}\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMax bin: ${maxBinCount} values`;

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
        module: comp.module,
        index: comp.index
    }));

    const tableConfig = {
        data: tableData,
        columns: [
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

    for (let i = 0; i < samplesToShow; i++) {
        const textHash = sampleHashes[i];
        const textSample = textSamples[textHash];

        if (!textSample) {
            console.warn(`Text sample not found for hash: ${textHash}`);
            continue;
        }

        // Get activations for this sample using full hash
        const fullHash = `${currentClusterHash}:${textHash}`;
        const activationIdx = activationsMap[fullHash];

        let tokenViz;
        if (activationIdx !== undefined && activationsArray) {
            // Get activation data from the NDArray
            const activations = activationsArray.get(activationIdx);
            const activationsData = Array.from(activations.data);

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
            console.warn(`No activations found for ${shortHash}`);
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