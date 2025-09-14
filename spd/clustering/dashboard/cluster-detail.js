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
    
    // Display components in dropdown
    const componentsSelect = document.getElementById('componentsSelect');
    const componentCount = document.getElementById('componentCount');
    
    componentCount.textContent = clusterData.components.length;
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
    
    // Display samples (up to 32)
    const tbody = document.getElementById('samplesTableBody');
    tbody.innerHTML = '';
    
    const samplesToShow = Math.min(32, clusterData.samples.length);
    
    for (let i = 0; i < samplesToShow; i++) {
        const sample = clusterData.samples[i];
        const tr = document.createElement('tr');
        
        // Create token visualization
        const tokenViz = document.createElement('div');
        tokenViz.className = 'token-container';
        
        sample.tokens.forEach((token, idx) => {
            const span = document.createElement('span');
            span.className = 'token';
            
            // Handle subword tokens
            if (token.startsWith('##')) {
                span.textContent = token.substring(2);
            } else {
                span.textContent = token;
            }
            
            // Color based on activation
            const activation = sample.activations[idx];
            const normalizedAct = Math.min(Math.max(activation, 0), 1);
            const intensity = Math.floor((1 - normalizedAct) * 255);
            span.style.backgroundColor = `rgb(255, ${intensity}, ${intensity})`;
            
            // Mark max position
            if (idx === sample.max_position) {
                span.style.border = '2px solid blue';
                span.style.fontWeight = 'bold';
            }
            
            // Add tooltip
            span.title = `${token}: ${activation.toFixed(6)}`;
            
            tokenViz.appendChild(span);
            
            // Add space after token (unless it's a subword)
            if (!sample.tokens[idx + 1]?.startsWith('##')) {
                tokenViz.appendChild(document.createTextNode(' '));
            }
        });
        
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