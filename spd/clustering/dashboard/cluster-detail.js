let clusterData = null;
let currentClusterId = null;

async function init() {
    // Get cluster ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    currentClusterId = urlParams.get('id');
    
    if (!currentClusterId) {
        document.getElementById('loading').textContent = 'No cluster ID specified';
        return;
    }
    
    await loadData();
}

async function loadData() {
    try {
        const response = await fetch('data/max_activations_iter7375_n16.json');
        const allData = await response.json();
        
        if (!allData[currentClusterId]) {
            document.getElementById('loading').textContent = 'Cluster not found';
            return;
        }
        
        clusterData = allData[currentClusterId];
        displayCluster();
        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        document.getElementById('loading').textContent = 'Error loading data: ' + error.message;
    }
}

function displayCluster() {
    // Update title
    document.getElementById('clusterTitle').textContent = `Cluster ${currentClusterId}`;
    
    // Display component count
    const componentCount = document.getElementById('componentCount');
    componentCount.textContent = clusterData.components.length;
    
    // Setup components table
    setupComponentsTable();
    
    // Display samples (up to 32)
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
        pageSize: 25,
        showFilters: true
    };

    new DataTable('#componentsTable', tableConfig);
}

function displaySamples() {
    const tbody = document.getElementById('samplesTableBody');
    tbody.innerHTML = '';
    
    const samplesToShow = Math.min(32, clusterData.samples.length);
    
    for (let i = 0; i < samplesToShow; i++) {
        const sample = clusterData.samples[i];
        const tr = document.createElement('tr');
        
        // Create token visualization with proper tooltips
        const tokenViz = createTokenVisualizationWithTooltip(
            sample.tokens, 
            sample.activations, 
            sample.max_position
        );
        
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${sample.dataset_index}</td>
            <td>${sample.max_activation.toFixed(4)}</td>
            <td>${sample.max_position}</td>
            <td>${sample.mean_activation.toFixed(4)}</td>
            <td></td>
        `;
        
        // Add token visualization to last cell
        tr.lastElementChild.appendChild(tokenViz);
        
        tbody.appendChild(tr);
    }
    
    if (clusterData.samples.length > 32) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="6" style="text-align: center;">
            ... and ${clusterData.samples.length - 32} more samples
        </td>`;
        tbody.appendChild(tr);
    }
}

// Initialize on page load
init();