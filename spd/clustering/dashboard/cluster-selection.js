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
        // Calculate statistics
        let maxActivation = 0;
        let sumMaxActivations = 0;
        
        cluster.samples.forEach(sample => {
            if (sample.max_activation > maxActivation) {
                maxActivation = sample.max_activation;
            }
            sumMaxActivations += sample.max_activation;
        });
        
        const meanMax = cluster.samples.length > 0 ? 
            sumMaxActivations / cluster.samples.length : 0;
        
        tableData.push({
            id: parseInt(clusterId),
            componentCount: cluster.components.length,
            sampleCount: cluster.samples.length,
            maxActivation: maxActivation,
            meanMax: meanMax
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
        4: (a, b) => a.meanMax - b.meanMax
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
        tr.innerHTML = `
            <td>${row.id}</td>
            <td>${row.componentCount}</td>
            <td>${row.sampleCount}</td>
            <td>${row.maxActivation.toFixed(4)}</td>
            <td>${row.meanMax.toFixed(4)}</td>
            <td><a href="cluster.html?id=${row.id}">View →</a></td>
        `;
        tbody.appendChild(tr);
    });
}

// Load data on page load
loadData();