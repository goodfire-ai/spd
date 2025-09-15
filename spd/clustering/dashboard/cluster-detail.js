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
        const response = await fetch('max_activations_iter-1_n4.json');
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
    
    // Setup components display
    setupComponentsDisplay();
    
    // Setup toggle button
    const toggleButton = document.getElementById('toggleComponents');
    const componentTable = document.getElementById('componentTable');
    const componentDropdown = document.getElementById('componentDropdown');
    
    // Show dropdown by default, but table if more than 5 components
    if (clusterData.components.length > 5) {
        toggleButton.textContent = 'Show Table';
        componentTable.style.display = 'none';
        componentDropdown.style.display = 'block';
    } else {
        toggleButton.textContent = 'Hide Table';
        componentTable.style.display = 'block';
        componentDropdown.style.display = 'none';
    }
    
    toggleButton.onclick = () => {
        if (componentTable.style.display === 'none') {
            componentTable.style.display = 'block';
            componentDropdown.style.display = 'none';
            toggleButton.textContent = 'Hide Table';
        } else {
            componentTable.style.display = 'none';
            componentDropdown.style.display = 'block';
            toggleButton.textContent = 'Show Table';
        }
    };
    
    // Display samples (up to 32)
    displaySamples();
}

function setupComponentsDisplay() {
    // Setup dropdown
    const componentsSelect = document.getElementById('componentsSelect');
    componentsSelect.innerHTML = '<option value="">Select a component...</option>';
    
    clusterData.components.forEach((comp, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${comp.module}:${comp.index}`;
        componentsSelect.appendChild(option);
    });
    
    componentsSelect.addEventListener('change', (e) => {
        if (e.target.value !== '') {
            const comp = clusterData.components[parseInt(e.target.value)];
            document.getElementById('componentDetails').innerHTML = `
                <div style="margin-top: 10px; padding: 10px; background: #f0f0f0;">
                    <strong>Module:</strong> ${comp.module}<br>
                    <strong>Index:</strong> ${comp.index}<br>
                    <strong>Label:</strong> ${comp.label}
                </div>
            `;
        } else {
            document.getElementById('componentDetails').innerHTML = '';
        }
    });
    
    // Setup table
    const columns = [
        { name: 'Module', sortable: true },
        { name: 'Index', sortable: true },
        { name: 'Label', sortable: true }
    ];
    
    const tableData = clusterData.components.map(comp => [
        comp.module,
        comp.index,
        comp.label
    ]);
    
    createSortableTable('componentsTable', columns, tableData, (row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row[0]}</td>
            <td>${row[1]}</td>
            <td>${row[2]}</td>
        `;
        return tr;
    });
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