let clusterData = null;
let currentClusterId = null;

// Create histogram bins from data
function createHistogramBins(data, numBins = 10) {
    if (!data || data.length === 0) {
        return [];
    }

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;

    if (range === 0) {
        return [data.length]; // All values are the same
    }

    const binWidth = range / numBins;
    const bins = Array(numBins).fill(0);

    // Fill bins
    data.forEach(value => {
        let binIndex = Math.floor((value - min) / binWidth);
        if (binIndex >= numBins) binIndex = numBins - 1;
        if (binIndex < 0) binIndex = 0;
        bins[binIndex]++;
    });

    return bins;
}

// Create activation histogram visualization
function createActivationHistogram(activations) {
    try {
        if (!activations || activations.length === 0) {
            return '<span style="color: #999; font-size: 11px;">No data</span>';
        }

        const container = document.createElement('div');
        container.className = 'sparkline-cell';
        container.style.width = '120px';
        container.style.height = '50px';
        container.style.display = 'flex';
        container.style.alignItems = 'center';
        container.style.justifyContent = 'center';

        // Create histogram bins (10 bins)
        const histogramCounts = createHistogramBins(activations, 10);

        // Use sparklines to render the histogram as a bar chart
        const svg = sparkbars(histogramCounts, null, {
            width: 120,
            height: 50,
            color: '#4169E1',
            shading: true, // Solid fill for histogram bars
            lineWidth: 0,  // No line, just bars
            markers: '',   // No markers
            margin: 2,
            ylims: [0, null],
            xAxis: {line: true, ticks: true, label_margin: 10},
            yAxis: {line: true, ticks: true, label_margin: 20}
        });

        container.innerHTML = svg;

        const min = Math.min(...activations);
        const max = Math.max(...activations);
        const mean = activations.reduce((a,b) => a+b, 0) / activations.length;
        const maxBinCount = Math.max(...histogramCounts);

        container.title = `Activation Histogram (n=${activations.length})\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMax bin: ${maxBinCount} samples`;

        return container;
    } catch (error) {
        console.warn('Error creating histogram:', error);
        return '<span style="color: #999; font-size: 11px;">Error</span>';
    }
}

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
    // TODO: Add activation histogram column for components
    // This requires backend changes to save component-level activation data
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

        // Create activation histogram
        const histogram = createActivationHistogram(sample.activations);

        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${sample.dataset_index}</td>
            <td>${sample.max_activation.toFixed(4)}</td>
            <td>${sample.max_position}</td>
            <td>${sample.mean_activation.toFixed(4)}</td>
            <td></td>
            <td></td>
        `;

        // Add histogram to second-to-last cell
        const histogramCell = tr.children[tr.children.length - 2];
        if (typeof histogram === 'string') {
            histogramCell.innerHTML = histogram;
        } else {
            histogramCell.appendChild(histogram);
        }

        // Add token visualization to last cell
        tr.lastElementChild.appendChild(tokenViz);
        
        tbody.appendChild(tr);
    }
    
    if (clusterData.samples.length > 32) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="7" style="text-align: center;">
            ... and ${clusterData.samples.length - 32} more samples
        </td>`;
        tbody.appendChild(tr);
    }
}

// Initialize on page load
init();