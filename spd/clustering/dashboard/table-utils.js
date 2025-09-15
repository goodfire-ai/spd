// Shared table utilities

function createSortableTable(tableId, columns, data, renderRowCallback) {
    const table = document.getElementById(tableId);
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    
    // Create header
    if (!thead) {
        const newThead = document.createElement('thead');
        table.appendChild(newThead);
    }
    
    const headerRow = document.createElement('tr');
    columns.forEach((col, index) => {
        const th = document.createElement('th');
        if (col.sortable) {
            th.onclick = () => sortTable(tableId, index, data, renderRowCallback);
            th.style.cursor = 'pointer';
            th.textContent = col.name + ' ↕';
        } else {
            th.textContent = col.name;
        }
        headerRow.appendChild(th);
    });
    
    thead.innerHTML = '';
    thead.appendChild(headerRow);
    
    // Render data
    renderTable(tableId, data, renderRowCallback);
}

function sortTable(tableId, columnIndex, data, renderRowCallback) {
    const table = document.getElementById(tableId);
    if (!table.sortState) {
        table.sortState = { column: -1, ascending: true };
    }
    
    // Toggle sort direction if same column
    if (table.sortState.column === columnIndex) {
        table.sortState.ascending = !table.sortState.ascending;
    } else {
        table.sortState.column = columnIndex;
        table.sortState.ascending = true;
    }
    
    // Sort data
    data.sort((a, b) => {
        let aVal = a[columnIndex];
        let bVal = b[columnIndex];
        
        // Handle different data types
        if (typeof aVal === 'number' && typeof bVal === 'number') {
            return aVal - bVal;
        } else {
            return String(aVal).localeCompare(String(bVal));
        }
    });
    
    if (!table.sortState.ascending) {
        data.reverse();
    }
    
    renderTable(tableId, data, renderRowCallback);
}

function renderTable(tableId, data, renderRowCallback) {
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    
    if (!tbody) {
        const newTbody = document.createElement('tbody');
        table.appendChild(newTbody);
    }
    
    tbody.innerHTML = '';
    
    data.forEach((row, index) => {
        const tr = renderRowCallback(row, index);
        tbody.appendChild(tr);
    });
}