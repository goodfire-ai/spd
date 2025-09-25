// Self-contained utilities for model visualization
// No global variables, all functions take necessary data as parameters

function getClusterModuleStats(clusterId, clusterData) {
    if (!clusterData || !clusterData[clusterId]) return {};

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

function renderModelArchitecture(clusterId, clusterData, modelInfo, colormap = 'blues') {
    if (!modelInfo || !modelInfo.module_list) {
        throw new Error('Model info not loaded');
    }

    const moduleStats = clusterData && clusterData[clusterId] ? getClusterModuleStats(clusterId, clusterData) : {};
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
        layer.attention.sort((a, b) => getModuleOrder(a.name) - getModuleOrder(b.name));
        layer.mlp.sort((a, b) => getModuleOrder(a.name) - getModuleOrder(b.name));
    });

    const sortedLayers = Object.keys(layerGroups).sort((a, b) => a - b);
    const cellSize = 12;

    const moduleElements = [];

    sortedLayers.forEach(layerNum => {
        const layer = layerGroups[layerNum];
        const layerElements = [];

        // Attention row (above MLP)
        if (layer.attention.length > 0) {
            const attentionRow = layer.attention.map(module => ({
                type: 'cell',
                module: module.name,
                count: module.count,
                components: module.components.map(c => c.index).join(','),
                color: getColorForValue(module.count, maxComponents, colormap),
                size: cellSize
            }));
            layerElements.push({ type: 'row', cells: attentionRow });
        }

        // MLP row (below attention)
        if (layer.mlp.length > 0) {
            const mlpRow = layer.mlp.map(module => ({
                type: 'cell',
                module: module.name,
                count: module.count,
                components: module.components.map(c => c.index).join(','),
                color: getColorForValue(module.count, maxComponents, colormap),
                size: cellSize
            }));
            layerElements.push({ type: 'row', cells: mlpRow });
        }

        // Other modules
        if (layer.other.length > 0) {
            const otherRow = layer.other.map(module => ({
                type: 'cell',
                module: module.name,
                count: module.count,
                components: module.components.map(c => c.index).join(','),
                color: getColorForValue(module.count, maxComponents, colormap),
                size: cellSize
            }));
            layerElements.push({ type: 'row', cells: otherRow });
        }

        if (layerElements.length > 0) {
            moduleElements.push({ type: 'layer', rows: layerElements });
        }
    });

    return {
        elements: moduleElements,
        maxComponents: maxComponents
    };
}

function renderToHTML(architecture) {
    let html = '';

    architecture.elements.forEach(layer => {
        html += '<div class="layer-block">';
        layer.rows.forEach(row => {
            html += '<div class="module-group">';
            row.cells.forEach(cell => {
                html += `<div class="module-cell" style="background-color: ${cell.color}; width: ${cell.size}px; height: ${cell.size}px;" data-module="${cell.module}" data-count="${cell.count}" data-components="${cell.components}"></div>`;
            });
            html += '</div>';
        });
        html += '</div>';
    });

    return html;
}

function setupTooltips(containerElement) {
    const tooltip = document.getElementById('tooltip');
    if (!tooltip) return;

    const cells = containerElement.querySelectorAll('.module-cell');

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