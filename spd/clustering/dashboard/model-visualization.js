let clusterData = {};
let modelInfo = {};
let currentClusterId = null;
let currentColormap = 'blues';

async function loadData(filename = 'max_activations_iter7375_n16.json') {
    try {
        const [clusterResponse, modelResponse] = await Promise.all([
            fetch('data/' + filename),
            fetch('data/model_info.json')
        ]);

        clusterData = await clusterResponse.json();
        modelInfo = await modelResponse.json();

        // Set default cluster to first available
        const clusterIds = Object.keys(clusterData);
        if (clusterIds.length > 0 && currentClusterId === null) {
            currentClusterId = clusterIds[0];
            document.getElementById('clusterInput').value = currentClusterId;
        }

        renderModelView();
        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        document.getElementById('loading').textContent = 'Error loading data: ' + error.message;
    }
}

function getClusterModuleStats(clusterId) {
    if (!clusterData[clusterId]) return {};

    const cluster = clusterData[clusterId];
    const moduleStats = {};

    // Count components per module for this specific cluster
    cluster.components.forEach(comp => {
        const module = comp.module;
        if (!moduleStats[module]) {
            moduleStats[module] = {
                componentCount: 0,
                components: []
            };
        }
        moduleStats[module].componentCount++;
        moduleStats[module].components.push(comp);
    });

    return moduleStats;
}

function getColorForValue(value, maxValue, colormap = 'blues') {
    if (maxValue === 0) return '#f5f5f5';
    const intensity = Math.min(value / maxValue, 1);

    const colormaps = {
        blues: {
            r: [247, 198, 107, 33],
            g: [251, 219, 174, 113],
            b: [255, 239, 214, 181]
        },
        reds: {
            r: [255, 252, 203, 103],
            g: [245, 174, 24, 0],
            b: [240, 145, 29, 13]
        },
        viridis: {
            r: [68, 59, 33, 253],
            g: [1, 82, 144, 231],
            b: [84, 139, 140, 37]
        },
        plasma: {
            r: [13, 126, 204, 240],
            g: [8, 3, 187, 249],
            b: [135, 192, 10, 33]
        }
    };

    const colors = colormaps[colormap] || colormaps.blues;
    const pos = intensity * (colors.r.length - 1);
    const i = Math.floor(pos);
    const f = pos - i;

    if (i >= colors.r.length - 1) {
        return `rgb(${colors.r[colors.r.length - 1]}, ${colors.g[colors.g.length - 1]}, ${colors.b[colors.b.length - 1]})`;
    }

    const r = Math.round(colors.r[i] + f * (colors.r[i + 1] - colors.r[i]));
    const g = Math.round(colors.g[i] + f * (colors.g[i + 1] - colors.g[i]));
    const b = Math.round(colors.b[i] + f * (colors.b[i + 1] - colors.b[i]));

    return `rgb(${r}, ${g}, ${b})`;
}

function getModuleOrder(moduleName) {
    if (moduleName.includes('q_proj')) return 0;
    if (moduleName.includes('k_proj')) return 1;
    if (moduleName.includes('v_proj')) return 2;
    if (moduleName.includes('o_proj')) return 3;
    if (moduleName.includes('gate_proj')) return 10;
    if (moduleName.includes('up_proj')) return 11;
    if (moduleName.includes('down_proj')) return 12;
    return 999;
}

function renderModelArchitecture(clusterId, totalWidth = 800, totalHeight = 400, colormap = 'blues') {
    if (!modelInfo.module_list) {
        return '<div>Model info not loaded</div>';
    }

    const moduleStats = clusterData[clusterId] ? getClusterModuleStats(clusterId) : {};
    const maxComponents = Math.max(...Object.values(moduleStats).map(s => s.componentCount), 1);

    // Group ALL modules from model_info by layer and type
    const layerGroups = {};

    modelInfo.module_list.forEach(moduleName => {
        const parts = moduleName.split('.');
        let layerNum = -1;
        let moduleType = 'other';

        for (let i = 0; i < parts.length; i++) {
            if (parts[i] === 'layers' && i + 1 < parts.length) {
                layerNum = parseInt(parts[i + 1]);
            }
        }

        if (moduleName.includes('self_attn')) {
            moduleType = 'attention';
        } else if (moduleName.includes('mlp')) {
            moduleType = 'mlp';
        }

        if (!layerGroups[layerNum]) {
            layerGroups[layerNum] = { attention: [], mlp: [], other: [] };
        }

        const count = moduleStats[moduleName] ? moduleStats[moduleName].componentCount : 0;
        const components = moduleStats[moduleName] ? moduleStats[moduleName].components : [];

        layerGroups[layerNum][moduleType].push({
            name: moduleName,
            count: count,
            components: components
        });
    });

    // Sort modules within each group by desired order
    Object.values(layerGroups).forEach(layer => {
        // Attention: q, k, v, o
        layer.attention.sort((a, b) => {
            const orderA = getModuleOrder(a.name);
            const orderB = getModuleOrder(b.name);
            return orderA - orderB;
        });

        // MLP: gate, up, down
        layer.mlp.sort((a, b) => {
            const orderA = getModuleOrder(a.name);
            const orderB = getModuleOrder(b.name);
            return orderA - orderB;
        });
    });

    const sortedLayers = Object.keys(layerGroups).sort((a, b) => a - b);
    const cellSize = 12;

    let html = '';

    sortedLayers.forEach(layerNum => {
        const layer = layerGroups[layerNum];

        html += `<div class="layer-block">`;

        // Attention row (above MLP)
        if (layer.attention.length > 0) {
            html += `<div class="module-group">`;
            layer.attention.forEach(module => {
                const color = getColorForValue(module.count, maxComponents, colormap);
                html += `<div class="module-cell" style="background-color: ${color}; width: ${cellSize}px; height: ${cellSize}px;" data-module="${module.name}" data-count="${module.count}" data-components="${module.components.map(c => c.index).join(',')}"></div>`;
            });
            html += `</div>`;
        }

        // MLP row (below attention)
        if (layer.mlp.length > 0) {
            html += `<div class="module-group">`;
            layer.mlp.forEach(module => {
                const color = getColorForValue(module.count, maxComponents, colormap);
                html += `<div class="module-cell" style="background-color: ${color}; width: ${cellSize}px; height: ${cellSize}px;" data-module="${module.name}" data-count="${module.count}" data-components="${module.components.map(c => c.index).join(',')}"></div>`;
            });
            html += `</div>`;
        }

        // Other modules
        if (layer.other.length > 0) {
            html += `<div class="module-group">`;
            layer.other.forEach(module => {
                const color = getColorForValue(module.count, maxComponents, colormap);
                html += `<div class="module-cell" style="background-color: ${color}; width: ${cellSize}px; height: ${cellSize}px;" data-module="${module.name}" data-count="${module.count}" data-components="${module.components.map(c => c.index).join(',')}"></div>`;
            });
            html += `</div>`;
        }

        html += `</div>`;
    });

    return html;
}

function renderModelView() {
    if (currentClusterId === null) return;

    const container = document.getElementById('modelContainer');
    const html = renderModelArchitecture(currentClusterId, 800, 400, currentColormap);
    container.innerHTML = html;

    // Add tooltip event listeners
    setupTooltips();

    // Update legend
    const moduleStats = getClusterModuleStats(currentClusterId);
    const maxComponents = Math.max(...Object.values(moduleStats).map(s => s.componentCount), 1);
    updateLegend(maxComponents);

    // Update cluster label
    document.getElementById('clusterLabel').textContent = currentClusterId;
}

function setupTooltips() {
    const tooltip = document.getElementById('tooltip');
    const cells = document.querySelectorAll('.module-cell');

    cells.forEach(cell => {
        cell.addEventListener('mouseenter', (e) => {
            const module = e.target.dataset.module;
            const count = e.target.dataset.count;
            const components = e.target.dataset.components;

            tooltip.textContent = `${module}\nComponents: ${count}\nIndices: ${components || 'none'}`;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.pageX + 10) + 'px';
            tooltip.style.top = (e.pageY + 10) + 'px';
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


function updateLegend(maxValue) {
    const legendScale = document.getElementById('legendScale');
    legendScale.innerHTML = '';

    // Create gradient
    for (let i = 0; i <= 100; i++) {
        const segment = document.createElement('div');
        segment.style.flex = '1';
        segment.style.backgroundColor = getColorForValue(i / 100 * maxValue, maxValue, currentColormap);
        legendScale.appendChild(segment);
    }

    // Update labels
    document.getElementById('minLabel').textContent = '0';
    document.getElementById('maxLabel').textContent = maxValue.toString();
}

// Event listeners
document.getElementById('clusterInput').addEventListener('change', (e) => {
    currentClusterId = e.target.value;
    renderModelView();
});

document.getElementById('dataFile').addEventListener('change', (e) => {
    document.getElementById('loading').style.display = 'block';
    loadData(e.target.value);
});

document.getElementById('colormapSelect').addEventListener('change', (e) => {
    currentColormap = e.target.value;
    renderModelView();
});

// Load initial data
loadData();