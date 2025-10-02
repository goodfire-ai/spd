let clusterData = {};
let modelInfo = {};
let dataTable = null;

// Custom column renderers
const columnRenderers = {
    modelView: function(value, row, col) {
        const clusterId = row.id;

        if (!modelInfo || !modelInfo.module_list) {
            return '<span style="color: #999; font-size: 11px;">Model info loading...</span>';
        }

        if (!clusterData[row.clusterHash]) {
            return '<span style="color: #999; font-size: 11px;">Cluster data missing</span>';
        }

        const architecture = renderModelArchitecture(row.clusterHash, clusterData, modelInfo, CONFIG.visualization.colormap);
        const html = renderToHTML(architecture);

        const container = document.createElement('div');
        container.className = 'model-view-cell';
        container.innerHTML = html;

        setTimeout(() => setupModelViewTooltips(container), 0);

        return container;
    },

    modulesSummary: function(value, row, col) {
        const modules = row.modules;
        const container = document.createElement('div');
        container.className = 'module-summary';

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
        const histData = row.stats.all_activations;
        if (!histData) {
            return '<span style="color: #999; font-size: 11px;">No data</span>';
        }

        const container = document.createElement('div');
        container.className = 'sparkline-cell';

        const svg = sparkbars(histData.bin_counts, null, {
            width: CONFIG.visualization.sparklineWidth,
            height: CONFIG.visualization.sparklineHeight,
            color: '#4169E1',
            shading: true,
            lineWidth: 0,
            markers: '',
            margin: 2,
            ylims: [0, null],
            logScale: true,
            xAxis: {line: true, ticks: true, label_margin: 10},
            yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin}
        });

        container.innerHTML = svg;

        const min = row.stats.min_activation;
        const max = row.stats.max_activation;
        const mean = row.stats.mean_activation;
        const n = row.stats.n_tokens;
        const maxBinCount = Math.max(...histData.bin_counts);

        container.title = `All Activations Histogram (n=${n})\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMean: ${mean.toFixed(4)}\nMax bin: ${maxBinCount} values`;

        return container;
    },

    maxActivationDistribution: function(value, row, col) {
        const histData = row.stats['max_activation-max-16'];
        if (!histData) {
            return '<span style="color: #999; font-size: 11px;">No data</span>';
        }

        const container = document.createElement('div');
        container.className = 'sparkline-cell';

        const svg = sparkbars(histData.bin_counts, null, {
            width: CONFIG.visualization.sparklineWidth,
            height: CONFIG.visualization.sparklineHeight,
            color: '#DC143C',
            shading: true,
            lineWidth: 0,
            markers: '',
            margin: 2,
            ylims: [0, null],
            logScale: true,
            xAxis: {line: true, ticks: true, label_margin: 10},
            yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin}
        });

        container.innerHTML = svg;

        const n = row.stats.n_samples;
        const maxBinCount = Math.max(...histData.bin_counts);
        const min = histData.bin_edges[0];
        const max = histData.bin_edges[histData.bin_edges.length - 1];

        container.title = `Max Activation Distribution (n=${n} samples)\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMax bin: ${maxBinCount} samples`;

        return container;
    },

    clusterLink: function(value, row, col) {
        return `<a href="cluster.html?id=${row.clusterHash}">View →</a>`;
    },

    tokenEntropy: function(value, row, col) {
        const tokenStats = row.stats.token_activations;
        if (!tokenStats) {
            return '<span style="color: #999; font-size: 11px;">N/A</span>';
        }
        return tokenStats.entropy.toFixed(2);
    },

    tokenConcentration: function(value, row, col) {
        const tokenStats = row.stats.token_activations;
        if (!tokenStats) {
            return '<span style="color: #999; font-size: 11px;">N/A</span>';
        }
        return (tokenStats.concentration_ratio * 100).toFixed(1) + '%';
    },

    topToken: function(value, row, col) {
        const tokenStats = row.stats.token_activations;
        if (!tokenStats || !tokenStats.top_tokens || tokenStats.top_tokens.length === 0) {
            return '<span style="color: #999; font-size: 11px;">N/A</span>';
        }

        const container = document.createElement('div');
        container.style.fontFamily = 'monospace';
        container.style.fontSize = '11px';
        container.style.lineHeight = '1.4';

        const topN = Math.min(5, tokenStats.top_tokens.length);
        const maxPercentage = tokenStats.top_tokens.length > 0
            ? ((tokenStats.top_tokens[0].count / tokenStats.total_activations) * 100)
            : 0;

        for (let i = 0; i < topN; i++) {
            const token = tokenStats.top_tokens[i];
            const tokenDisplay = token.token.replace(/ /g, '·').replace(/\n/g, '↵');
            const percentageValue = ((token.count / tokenStats.total_activations) * 100);
            const percentage = percentageValue.toFixed(1);

            // Color based on percentage (normalized by max percentage)
            const normalizedPct = maxPercentage > 0 ? percentageValue / maxPercentage : 0;
            const intensity = Math.floor((1 - normalizedPct) * 255);
            const bgColor = `rgb(255, ${intensity}, ${intensity})`;

            const line = document.createElement('div');
            line.style.display = 'flex';
            line.style.justifyContent = 'space-between';
            line.style.gap = '8px';

            const tokenSpan = document.createElement('span');
            tokenSpan.innerHTML = `<code class="token-display">${tokenDisplay}</code>`;
            tokenSpan.style.textAlign = 'left';

            const pctSpan = document.createElement('span');
            pctSpan.textContent = `${percentage}%`;
            pctSpan.style.textAlign = 'right';
            pctSpan.style.backgroundColor = bgColor;
            pctSpan.style.padding = '2px 4px';
            pctSpan.style.borderRadius = '2px';

            line.appendChild(tokenSpan);
            line.appendChild(pctSpan);
            container.appendChild(line);
        }

        return container;
    },

    // Generic histogram renderer for any BinnedData stat
    genericHistogram: function(statKey, color, title) {
        return function(value, row, col) {
            const histData = row.stats[statKey];
            if (!histData || !histData.bin_counts) {
                return '<span style="color: #999; font-size: 11px;">No data</span>';
            }

            const container = document.createElement('div');
            container.className = 'sparkline-cell';

            const svg = sparkbars(histData.bin_counts, null, {
                width: CONFIG.visualization.sparklineWidth,
                height: CONFIG.visualization.sparklineHeight,
                color: color,
                shading: true,
                lineWidth: 0,
                markers: '',
                margin: 2,
                ylims: [0, null],
                logScale: true,
                xAxis: {line: true, ticks: true, label_margin: 10},
                yAxis: {line: true, ticks: true, label_margin: CONFIG.visualization.sparklineYAxisMargin}
            });

            container.innerHTML = svg;

            const maxBinCount = Math.max(...histData.bin_counts);
            const min = histData.bin_edges[0];
            const max = histData.bin_edges[histData.bin_edges.length - 1];

            container.title = `${title}\n\nMin: ${min.toFixed(4)}\nMax: ${max.toFixed(4)}\nMax bin: ${maxBinCount} values`;

            return container;
        };
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
    const response = await fetch(CONFIG.getDataPath('modelInfo'));
    modelInfo = await response.json();
    displayModelInfo();
}

function displayModelInfo() {
    const modelInfoDiv = document.getElementById('modelInfo');
    if (Object.keys(modelInfo).length > 0) {
        document.getElementById('totalModules').textContent = modelInfo.total_modules;
        document.getElementById('totalComponents').textContent = modelInfo.total_components;
        document.getElementById('totalClusters').textContent = modelInfo.total_clusters;

        const totalParams = modelInfo.total_parameters;
        const formatted = totalParams >= 1000000
            ? (totalParams / 1000000).toFixed(1) + 'M'
            : totalParams >= 1000
            ? (totalParams / 1000).toFixed(1) + 'K'
            : totalParams.toString();
        document.getElementById('totalParameters').textContent = formatted;

        modelInfoDiv.style.display = 'block';
    }
}

function processClusterData() {
    const tableData = [];

    for (const [clusterHash, cluster] of Object.entries(clusterData)) {
        const modules = new Set();
        cluster.components.forEach(comp => {
            modules.add(comp.module);
        });

        const stats = cluster.stats;

        // Extract cluster ID from hash (format: "runid-iteration-clusteridx")
        const parts = clusterHash.split('-');
        const clusterId = parseInt(parts[parts.length - 1]);

        tableData.push({
            id: clusterId,
            clusterHash: clusterHash,
            componentCount: cluster.components.length,
            modules: Array.from(modules),
            stats: stats
        });
    }

    return tableData;
}

async function loadData() {
    const [clusterResponse] = await Promise.all([
        fetch(CONFIG.getDataPath('clusters')),
        loadModelInfo()
    ]);

    clusterData = await clusterResponse.json();

    const tableData = processClusterData();

    // Discover histogram stats from first cluster
    const firstCluster = Object.values(clusterData)[0];
    const histogramStats = [];
    if (firstCluster && firstCluster.stats) {
        for (const [key, value] of Object.entries(firstCluster.stats)) {
            if (value && typeof value === 'object' && 'bin_counts' in value && 'bin_edges' in value) {
                histogramStats.push(key);
            }
        }
    }

    // Base columns
    const columns = [
        {
            key: 'id',
            label: 'ID',
            type: 'number',
            width: '10px',
            align: 'center'
        },
        {
            key: 'componentCount',
            label: 'Comps',
            type: 'number',
            width: '10px',
            align: 'right'
        },
        {
            key: 'componentCount',
            label: 'Model View',
            type: 'number',
            width: '21px',
            align: 'center',
            renderer: columnRenderers.modelView
        },
        {
            key: 'modules',
            label: 'Modules',
            type: 'string',
            width: '10px',
            renderer: columnRenderers.modulesSummary
        }
    ];

    // Add histogram columns dynamically
    const statColors = {
        'all_activations': '#4169E1',
        'max_activation-max-16': '#DC143C',
        'max_activation-max-32': '#DC143C',
        'mean_activation-max-16': '#228B22',
        'median_activation-max-16': '#FF8C00',
        'min_activation-max-16': '#9370DB',
        'max_activation_position': '#FF6347'
    };

    histogramStats.forEach(statKey => {
        const color = statColors[statKey] || '#808080';
        const label = statKey.replace(/-/g, ' ').replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');

        columns.push({
            key: 'stats',
            label: label,
            type: 'string',
            width: '200px',
            align: 'center',
            renderer: columnRenderers.genericHistogram(statKey, color, label)
        });
    });

    // Token activation columns
    columns.push({
        key: 'stats',
        label: 'Top Tokens',
        type: 'string',
        width: '150px',
        align: 'left',
        renderer: columnRenderers.topToken
    });

    columns.push({
        key: 'stats',
        label: 'Token Entropy',
        type: 'number',
        width: '60px',
        align: 'right',
        renderer: columnRenderers.tokenEntropy
    });

    columns.push({
        key: 'stats',
        label: 'Token Conc.',
        type: 'string',
        width: '60px',
        align: 'right',
        renderer: columnRenderers.tokenConcentration
    });

    // Actions column
    columns.push({
        key: 'id',
        label: 'Actions',
        type: 'string',
        width: '20px',
        align: 'center',
        renderer: columnRenderers.clusterLink
    });

    const tableConfig = {
        data: tableData,
        columns: columns,
        pageSize: CONFIG.indexPage.pageSize,
        pageSizeOptions: CONFIG.indexPage.pageSizeOptions,
        showFilters: CONFIG.indexPage.showFilters
    };

    dataTable = new DataTable('#clusterTableContainer', tableConfig);

    document.getElementById('loading').style.display = 'none';
}

document.addEventListener('DOMContentLoaded', async () => {
    await initConfig();
    loadData();
});
