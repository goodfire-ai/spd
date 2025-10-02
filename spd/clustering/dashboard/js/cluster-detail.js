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
    try {
        // Load JSON data in parallel
        console.log('Loading JSON files...');
        const [clustersResponse, textSamplesResponse, activationsMapResponse, modelInfoResponse] = await Promise.all([
            fetch(CONFIG.getDataPath('clusters')),
            fetch(CONFIG.getDataPath('textSamples')),
            fetch(CONFIG.getDataPath('activationsMap')),
            fetch(CONFIG.getDataPath('modelInfo'))
        ]);

        if (!clustersResponse.ok) {
            throw new Error(`Failed to load clusters: ${clustersResponse.status}`);
        }
        if (!textSamplesResponse.ok) {
            throw new Error(`Failed to load text samples: ${textSamplesResponse.status}`);
        }
        if (!activationsMapResponse.ok) {
            throw new Error(`Failed to load activations map: ${activationsMapResponse.status}`);
        }
        if (!modelInfoResponse.ok) {
            throw new Error(`Failed to load model info: ${modelInfoResponse.status}`);
        }

        console.log('Parsing clusters...');
        allClusters = await clustersResponse.json();

        console.log('Parsing text samples...');
        textSamples = await textSamplesResponse.json();

        console.log('Parsing activations map...');
        activationsMap = await activationsMapResponse.json();

        console.log('Parsing model info...');
        modelInfo = await modelInfoResponse.json();

        if (!allClusters[currentClusterHash]) {
            document.getElementById('loading').textContent = 'Cluster not found';
            return;
        }

        clusterData = allClusters[currentClusterHash];

        // Load activations (float16 compressed npz)
        const activationsPath = CONFIG.getDataPath('activations');
        console.log('Loading activations array from:', activationsPath);
        activationsArray = await NDArray.load(activationsPath);
        console.log('Activations loaded:', activationsArray.shape);

        displayCluster();
        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        document.getElementById('loading').textContent = 'Error loading data: ' + error.message;
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
    if (!modelInfo || !modelInfo.module_list) {
        console.warn('Model info not loaded');
        return;
    }

    const modelViewDiv = document.getElementById('modelView');

    try {
        const architecture = renderModelArchitecture(currentClusterHash, allClusters, modelInfo, CONFIG.visualization.colormap);
        const html = renderToHTML(architecture);
        modelViewDiv.innerHTML = html;

        // Setup tooltips after rendering
        setTimeout(() => setupTooltips(modelViewDiv), 0);
    } catch (error) {
        console.error('Failed to render model visualization:', error);
        modelViewDiv.innerHTML = '<span style="color: #999; font-size: 11px;">Model visualization unavailable</span>';
    }
}

function displayTokenActivations() {
    const tokenStats = clusterData.stats.token_activations;

    // Show the section
    document.getElementById('tokenActivations').style.display = 'block';

    // Populate summary statistics
    document.getElementById('totalUniqueTokens').textContent =
        tokenStats.total_unique_tokens.toLocaleString();
    document.getElementById('totalActivations').textContent =
        tokenStats.total_activations.toLocaleString();
    document.getElementById('entropy').textContent =
        tokenStats.entropy.toFixed(2);
    document.getElementById('concentrationRatio').textContent =
        (tokenStats.concentration_ratio * 100).toFixed(1) + '%';

    // Setup top tokens table
    if (tokenStats.top_tokens && tokenStats.top_tokens.length > 0) {
        const tableData = tokenStats.top_tokens.map((item, idx) => ({
            rank: idx + 1,
            token: item.token,
            count: item.count,
            percentage: ((item.count / tokenStats.total_activations) * 100).toFixed(1)
        }));

        const tableConfig = {
            data: tableData,
            columns: [
                {
                    key: 'rank',
                    label: 'Rank',
                    type: 'number',
                    width: '60px',
                    align: 'right'
                },
                {
                    key: 'token',
                    label: 'Token',
                    type: 'string',
                    width: '200px',
                    render: (value) => {
                        // Show token in a monospace box with visual formatting
                        const tokenDisplay = value.replace(/ /g, '·').replace(/\n/g, '↵');
                        return `<code class="token-display">${tokenDisplay}</code>`;
                    }
                },
                {
                    key: 'count',
                    label: 'Count',
                    type: 'number',
                    width: '100px',
                    align: 'right',
                    render: (value) => value.toLocaleString()
                },
                {
                    key: 'percentage',
                    label: '%',
                    type: 'number',
                    width: '80px',
                    align: 'right',
                    render: (value) => value + '%'
                }
            ],
            pageSize: 20,
            showFilters: false
        };

        new DataTable('#topTokensTable', tableConfig);
    }
}

function setupComponentsTable() {
    const tableData = clusterData.components.map(comp => ({
        module: comp.module,
        index: comp.index,
        label: comp.label
    }));

    const tableConfig = {
        data: tableData,
        columns: [
            {
                key: 'module',
                label: 'Module',
                type: 'string',
                width: '200px'
            },
            {
                key: 'index',
                label: 'Index',
                type: 'number',
                width: '80px',
                align: 'right'
            },
            {
                key: 'label',
                label: 'Label',
                type: 'string',
                width: '300px'
            }
        ],
        pageSize: CONFIG.clusterPage.pageSize,
        showFilters: CONFIG.clusterPage.showFilters
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

        // Get activations for this sample
        // Extract just clusterLabel from full hash (format: "runid-iteration-clusterLabel")
        const clusterLabel = currentClusterHash.split('-').pop();
        const shortHash = `${clusterLabel}:${textHash}`;
        const activationIdx = activationsMap[shortHash];

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