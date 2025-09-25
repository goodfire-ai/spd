let clusterData = {};
let tableData = [];
let modelInfo = {};
let currentSort = { column: 1, ascending: false }; // Default sort by components descending

async function loadModelInfo() {
    try {
        const response = await fetch('model_info.json');
        modelInfo = await response.json();
        displayModelInfo();
    } catch (error) {
        console.warn('Could not load model info:', error.message);
        // Don't show error to user, model info is optional
    }
}

function displayModelInfo() {
    const modelInfoDiv = document.getElementById('modelInfo');
    if (Object.keys(modelInfo).length > 0) {
        document.getElementById('totalModules').textContent = modelInfo.total_modules || '-';
        document.getElementById('totalComponents').textContent = modelInfo.total_components || '-';
        document.getElementById('totalClusters').textContent = modelInfo.total_clusters || '-';

        // Format parameter count
        const totalParams = modelInfo.total_parameters;
        if (totalParams) {
            const formatted = totalParams >= 1000000
                ? (totalParams / 1000000).toFixed(1) + 'M'
                : totalParams >= 1000
                ? (totalParams / 1000).toFixed(1) + 'K'
                : totalParams.toString();
            document.getElementById('totalParameters').textContent = formatted;
        } else {
            document.getElementById('totalParameters').textContent = '-';
        }

        modelInfoDiv.style.display = 'block';
    }
}

async function loadData() {
    try {
        // Load cluster data and model info in parallel
        const [clusterResponse] = await Promise.all([
            fetch('max_activations_iter7375_n16.json'),
            loadModelInfo()
        ]);

        clusterData = await clusterResponse.json();
        processTableData();
        renderTable();
        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        document.getElementById('loading').textContent = 'Error loading data: ' + error.message;
    }
}

function processTableData() {
    tableData = [];
    
    for (const [clusterId, cluster] of Object.entries(clusterData)) {
        // Get unique modules
        const modules = new Set();
        cluster.components.forEach(comp => {
            const module = comp.module;
            modules.add(module);
        });
        
        // Calculate statistics from all activations
        const allActivations = [];
        cluster.samples.forEach(sample => {
            sample.activations.forEach(act => {
                if (act > 0) { // Only consider non-zero activations
                    allActivations.push(act);
                }
            });
        });
        
        // Calculate stats
        let maxActivation = 0;
        let minActivation = Infinity;
        let sumActivations = 0;
        
        if (allActivations.length > 0) {
            allActivations.sort((a, b) => a - b);
            maxActivation = allActivations[allActivations.length - 1];
            minActivation = allActivations[0];
            sumActivations = allActivations.reduce((a, b) => a + b, 0);
        } else {
            minActivation = 0;
        }
        
        const meanActivation = allActivations.length > 0 ? 
            sumActivations / allActivations.length : 0;
        
        const medianActivation = allActivations.length > 0 ?
            (allActivations.length % 2 === 0 ?
                (allActivations[Math.floor(allActivations.length / 2) - 1] + 
                 allActivations[Math.floor(allActivations.length / 2)]) / 2 :
                allActivations[Math.floor(allActivations.length / 2)]) : 0;
        
        tableData.push({
            id: parseInt(clusterId),
            componentCount: cluster.components.length,
            modules: Array.from(modules),
            sampleCount: cluster.samples.length,
            maxActivation: maxActivation,
            meanActivation: meanActivation,
            medianActivation: medianActivation,
            minActivation: minActivation
        });
    }
    
    // Initial sort by component count
    sortTableData(1, false);
}

function sortTableData(column, ascending) {
    const sortFunctions = {
        0: (a, b) => a.id - b.id,
        1: (a, b) => a.componentCount - b.componentCount,
        2: (a, b) => a.sampleCount - b.sampleCount,
        3: (a, b) => a.maxActivation - b.maxActivation,
        4: (a, b) => a.meanActivation - b.meanActivation,
        5: (a, b) => a.medianActivation - b.medianActivation,
        6: (a, b) => a.minActivation - b.minActivation
    };
    
    tableData.sort(sortFunctions[column]);
    
    if (!ascending) {
        tableData.reverse();
    }
}

function sortTable(column) {
    // Toggle sort direction if same column
    if (currentSort.column === column) {
        currentSort.ascending = !currentSort.ascending;
    } else {
        currentSort.column = column;
        currentSort.ascending = true;
    }
    
    sortTableData(column, currentSort.ascending);
    renderTable();
}

function renderTable() {
    const tbody = document.getElementById('clusterTableBody');
    tbody.innerHTML = '';
    
    tableData.forEach(row => {
        const tr = document.createElement('tr');
        
        // Format modules display
        let modulesDisplay;
        if (row.modules.length === 1) {
            // Show the single module name (shortened)
            const parts = row.modules[0].split('.');
            modulesDisplay = parts.length > 2 ? parts.slice(-2).join('.') : row.modules[0];
        } else {
            // Show count for multiple modules
            modulesDisplay = `${row.modules.length} modules`;
        }
        
        tr.innerHTML = `
            <td>${row.id}</td>
            <td>${row.componentCount}</td>
            <td title="${row.modules.join('\n')}">${modulesDisplay}</td>
            <td>${row.sampleCount}</td>
            <td>${row.maxActivation.toFixed(4)}</td>
            <td>${row.meanActivation.toFixed(4)}</td>
            <td>${row.medianActivation.toFixed(4)}</td>
            <td>${row.minActivation.toFixed(6)}</td>
            <td><a href="cluster.html?id=${row.id}">View →</a></td>
        `;
        tbody.appendChild(tr);
    });
}

// Load data on page load
loadData();