let clusterData = {};
let modelInfo = {};
let dataTable = null;

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

// Custom column renderers
const columnRenderers = {
    modelView: function(value, row, col) {
        const clusterId = row.id;

        // Debug logging
        console.log('Rendering model view for cluster', clusterId);
        console.log('clusterData keys:', Object.keys(clusterData));
        console.log('modelInfo:', modelInfo);

        if (!modelInfo || !modelInfo.module_list) {
            console.warn('Model info not available');
            return '<span style="color: #999; font-size: 11px;">Model info loading...</span>';
        }

        if (!clusterData[clusterId]) {
            console.warn('Cluster data not found for', clusterId);
            return '<span style="color: #999; font-size: 11px;">Cluster data missing</span>';
        }

        try {
            // Create compact model architecture visualization
            const architecture = renderModelArchitecture(clusterId, clusterData, modelInfo, 'blues');
            const html = renderToHTML(architecture);

            const container = document.createElement('div');
            container.className = 'model-view-cell';
            container.innerHTML = html;

            // Add tooltip functionality
            setTimeout(() => setupModelViewTooltips(container), 0);

            return container;
        } catch (error) {
            console.error('Error rendering model view for cluster', clusterId, error);
            return `<span style="color: #999; font-size: 11px;">Error: ${error.message}</span>`;
        }
    },

    modulesSummary: function(value, row, col) {
        const modules = row.modules || [];
        const container = document.createElement('div');
        container.className = 'module-summary';

        if (modules.length === 0) {
            container.textContent = 'No modules';
            return container;
        }

        if (modules.length === 1) {
            const parts = modules[0].split('.');
            container.textContent = parts.length > 2 ? parts.slice(-2).join('.') : modules[0];
        } else if (modules.length <= 3) {
            container.textContent = modules.map(m => {
                const parts = m.split('.');
                return parts.length > 2 ? parts.slice(-2).join('.') : m;
            }).join(', ');
        } else {
            container.textContent = `${modules.length} modules`;
        }

        container.title = modules.join('\n');
        return container;
    },

    activationHistogram: function(value, row, col) {
        try {
            const activations = row.allActivations || [];
            if (activations.length === 0) {
                return '<span style="color: #999; font-size: 11px;">No data</span>';
            }

            const container = document.createElement('div');
            container.className = 'sparkline-cell';

            // Create histogram bins (10 bins)
            const histogramCounts = createHistogramBins(activations, 10);

            // Use sparklines to render the histogram as a bar chart
            const svg = sparkline(histogramCounts, null, {
                width: 120,
                height: 30,
                color: '#4169E1',
                shading: true, // Solid fill for histogram bars
                lineWidth: 0,  // No line, just bars
                markers: '',   // No markers
                margin: 2,
                xAxis: {line: true, ticks: true},
                yAxis: {line: true, ticks: true}
            });

            container.innerHTML = svg;

            const min = Math.min(...activations);
            const max = Math.max(...activations);
            const mean = activations.reduce((a,b) => a+b, 0) / activations.length;
            const maxBinCount = Math.max(...histogramCounts);

            container.title = `Activation Histogram (n=${activations.length})\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMax bin: ${maxBinCount} samples`;

            return container;
        } catch (error) {
            console.warn('Error creating histogram for cluster', row.id, error);
            return '<span style="color: #999; font-size: 11px;">Error</span>';
        }
    },

    clusterLink: function(value, row, col) {
        return `<a href="cluster.html?id=${row.id}">View →</a>`;
    }
};

function setupModelViewTooltips(container) {
    const tooltip = document.getElementById('tooltip');
    if (!tooltip) return;

    const cells = container.querySelectorAll('.module-cell');

    cells.forEach(cell => {
        cell.addEventListener('mouseenter', (e) => {
            const module = e.target.dataset.module;
            const count = e.target.dataset.count;
            const components = e.target.dataset.components;

            if (module) {
                tooltip.textContent = `${module}\nComponents: ${count}\nIndices: ${components || 'none'}`;
                tooltip.style.display = 'block';
                tooltip.style.left = (e.pageX + 10) + 'px';
                tooltip.style.top = (e.pageY + 10) + 'px';
            }
        });

        cell.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });

        cell.addEventListener('mousemove', (e) => {
            tooltip.style.left = (e.pageX + 10) + 'px';
            tooltip.style.top = (e.pageY + 10) + 'px';
        });
    });
}

async function loadModelInfo() {
    try {
        const response = await fetch('data/model_info.json');
        modelInfo = await response.json();
        displayModelInfo();
    } catch (error) {
        console.warn('Could not load model info:', error.message);
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

function processClusterData() {
    const tableData = [];

    for (const [clusterId, cluster] of Object.entries(clusterData)) {
        // Get unique modules
        const modules = new Set();
        cluster.components.forEach(comp => {
            modules.add(comp.module);
        });

        // Calculate activation statistics
        const allActivations = [];
        cluster.samples.forEach(sample => {
            sample.activations.forEach(act => {
                if (act > 0) {
                    allActivations.push(act);
                }
            });
        });

        // Calculate stats
        let maxActivation = 0;
        let minActivation = Infinity;
        let meanActivation = 0;
        let medianActivation = 0;

        if (allActivations.length > 0) {
            const sorted = [...allActivations].sort((a, b) => a - b);
            maxActivation = sorted[sorted.length - 1];
            minActivation = sorted[0];
            meanActivation = allActivations.reduce((a, b) => a + b, 0) / allActivations.length;
            medianActivation = sorted.length % 2 === 0
                ? (sorted[Math.floor(sorted.length / 2) - 1] + sorted[Math.floor(sorted.length / 2)]) / 2
                : sorted[Math.floor(sorted.length / 2)];
        } else {
            minActivation = 0;
        }

        tableData.push({
            id: parseInt(clusterId),
            componentCount: cluster.components.length,
            modules: Array.from(modules),
            sampleCount: cluster.samples.length,
            maxActivation: maxActivation,
            meanActivation: meanActivation,
            medianActivation: medianActivation,
            minActivation: minActivation,
            allActivations: allActivations
        });
    }

    return tableData;
}

async function loadData() {
    try {
        // Load cluster data and model info in parallel
        const [clusterResponse] = await Promise.all([
            fetch('data/max_activations_iter7375_n16.json'),
            loadModelInfo()
        ]);

        clusterData = await clusterResponse.json();

        // Process data for table
        const tableData = processClusterData();

        // Configure and create DataTable
        const tableConfig = {
            data: tableData,
            columns: [
                {
                    key: 'id',
                    label: 'ID',
                    type: 'number',
                    width: '10px',
                    align: 'center'
                },
                {
                    key: 'componentCount',
                    label: 'Model View',
                    type: 'number',
                    width: '160px',
                    align: 'center',
                    renderer: columnRenderers.modelView
                },
                {
                    key: 'componentCount',
                    label: 'Comps',
                    type: 'number',
                    width: '50px',
                    align: 'right'
                },
                {
                    key: 'modules',
                    label: 'Modules',
                    type: 'string',
                    width: '100px',
                    renderer: columnRenderers.modulesSummary
                },
                {
                    key: 'allActivations',
                    label: 'Activations',
                    type: 'string',
                    width: '100px',
                    align: 'center',
                    renderer: columnRenderers.activationHistogram
                },
                // {
                //     key: 'sampleCount',
                //     label: 'Samples',
                //     type: 'number',
                //     width: '50px',
                //     align: 'right'
                // },
                // {
                //     key: 'maxActivation',
                //     label: 'Max',
                //     type: 'number',
                //     width: '20px',
                //     align: 'right',
                //     renderer: (value) => value.toFixed(3)
                // },
                // {
                //     key: 'meanActivation',
                //     label: 'Mean',
                //     type: 'number',
                //     width: '20px',
                //     align: 'right',
                //     renderer: (value) => value.toFixed(3)
                // },
                // {
                //     key: 'medianActivation',
                //     label: 'Med',
                //     type: 'number',
                //     width: '20px',
                //     align: 'right',
                //     renderer: (value) => value.toFixed(3)
                // },
                // {
                //     key: 'minActivation',
                //     label: 'Min',
                //     type: 'number',
                //     width: '20px',
                //     align: 'right',
                //     renderer: (value) => value.toFixed(4)
                // },
                {
                    key: 'id',
                    label: 'Actions',
                    type: 'string',
                    width: '20px',
                    align: 'center',
                    renderer: columnRenderers.clusterLink
                }
            ],
            pageSize: 25,
            pageSizeOptions: [10, 25, 50, 100],
            showFilters: true
        };

        // Create table
        dataTable = new DataTable('#clusterTableContainer', tableConfig);

        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        document.getElementById('loading').textContent = 'Error loading data: ' + error.message;
        console.error('Error loading data:', error);
    }
}

// Load data on page load
document.addEventListener('DOMContentLoaded', loadData);