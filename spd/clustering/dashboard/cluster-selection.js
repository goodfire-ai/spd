let clusterData = {};
let tableData = [];
let currentSort = { column: 1, ascending: false }; // Default sort by components descending

async function loadData() {
    try {
        const response = await fetch('max_activations_iter-1_n4.json');
        clusterData = await response.json();
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
        
        // Format modules list - show shortened module names
        const modulesList = row.modules.map(m => {
            // Extract last two parts of module path for brevity
            const parts = m.split('.');
            if (parts.length > 2) {
                return parts.slice(-2).join('.');
            }
            return m;
        }).join(', ');
        
        tr.innerHTML = `
            <td>${row.id}</td>
            <td>${row.componentCount}</td>
            <td title="${row.modules.join('\n')}">${modulesList}</td>
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