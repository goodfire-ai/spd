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