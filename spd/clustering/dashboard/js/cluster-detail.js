let clusterData = null;
let textSamples = {};
let activationsArray = null;
let activationsMap = {};
let currentClusterHash = null;

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
        const [clustersResponse, textSamplesResponse, activationsMapResponse] = await Promise.all([
            fetch(CONFIG.data.clusterDataFile),
            fetch(CONFIG.data.textSamplesFile),
            fetch(CONFIG.data.activationsMapFile)
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

        console.log('Parsing clusters...');
        const allClusters = await clustersResponse.json();

        console.log('Parsing text samples...');
        textSamples = await textSamplesResponse.json();

        console.log('Parsing activations map...');
        activationsMap = await activationsMapResponse.json();

        if (!allClusters[currentClusterHash]) {
            document.getElementById('loading').textContent = 'Cluster not found';
            return;
        }

        clusterData = allClusters[currentClusterHash];

        // Load activations .npy file
        console.log('Loading activations array from:', CONFIG.data.activationsFile);
        activationsArray = await NDArray.load(CONFIG.data.activationsFile);
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

    // Setup components table
    setupComponentsTable();

    // Display samples
    displaySamples();
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
        const activationHash = `${currentClusterHash}:${textHash}`;
        const activationIdx = activationsMap[activationHash];

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
            console.warn(`No activations found for ${activationHash}`);
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