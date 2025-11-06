// tree-viewer.js - Simple decision tree viewer

// Parse URL parameters
function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const label = params.get('label');
    if (!label) {
        return { label: null, module: null, component: null };
    }

    // Split on last colon to get module and component
    const lastColon = label.lastIndexOf(':');
    if (lastColon === -1) {
        throw new Error(`Invalid label format: ${label}. Expected format: module:component`);
    }

    const module = label.substring(0, lastColon);
    const component = parseInt(label.substring(lastColon + 1));

    return { label, module, component };
}

// Load trees.json
async function loadTreeData() {
    const response = await fetch('data/trees.json');
    if (!response.ok) {
        throw new Error('Failed to load trees.json');
    }
    return await response.json();
}

// Find the layer and tree for the target component
function findComponentTree(treesData, targetModule, targetComponent) {
    // Find the layer with this module
    for (const layerData of treesData) {
        if (layerData.module_key === targetModule) {
            // Find the index in varying_component_indices
            const varyingIndices = layerData.varying_component_indices;
            const treeIndex = varyingIndices.indexOf(targetComponent);

            if (treeIndex === -1) {
                throw new Error(`Component ${targetComponent} not found in module ${targetModule}'s varying components`);
            }

            return {
                layerData: layerData,
                tree: layerData.trees[treeIndex],
                treeIndex: treeIndex
            };
        }
    }
    throw new Error(`Module ${targetModule} not found in tree data`);
}

// Render a single tree node recursively
function renderTreeNode(tree, nodeIdx, featureMap, depth = 0) {
    const feature = tree.feature[nodeIdx];
    const threshold = tree.threshold[nodeIdx];
    const nSamples = tree.n_node_samples[nodeIdx];
    const value = tree.value[nodeIdx];

    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'tree-node';
    nodeDiv.style.marginLeft = `${depth * 20}px`;

    // Check if leaf node
    if (feature === -2) {
        nodeDiv.classList.add('leaf');
        // For binary classification, value is [[n_class_0, n_class_1]]
        const class0 = value[0][0];
        const class1 = value[0][1];
        const prediction = class1 > class0 ? 1 : 0;
        nodeDiv.innerHTML = `
            <strong>Leaf:</strong> Predict ${prediction}
            <div class="node-samples">${nSamples} samples (0: ${class0.toFixed(1)}, 1: ${class1.toFixed(1)})</div>
        `;
        return nodeDiv;
    }

    // Split node
    const componentInfo = featureMap[feature];
    if (!componentInfo) {
        nodeDiv.innerHTML = `<strong>ERROR:</strong> Invalid feature index ${feature}`;
        return nodeDiv;
    }

    // Construct label from module_key and component_idx
    const label = `${componentInfo.module_key}:${componentInfo.component_idx}`;
    const componentLink = `component.html?label=${encodeURIComponent(label)}`;

    nodeDiv.innerHTML = `
        <div class="node-samples">${nSamples} samples</div>
        <strong>If</strong> <a href="${componentLink}" class="component-link">${label}</a>
    `;

    // Render children
    const childrenDiv = document.createElement('div');
    childrenDiv.className = 'tree-children';

    // Left child (false branch)
    const leftLabel = document.createElement('div');
    leftLabel.innerHTML = '<strong>False:</strong>';
    childrenDiv.appendChild(leftLabel);
    childrenDiv.appendChild(renderTreeNode(tree, tree.children_left[nodeIdx], featureMap, depth + 1));

    // Right child (true branch)
    const rightLabel = document.createElement('div');
    rightLabel.innerHTML = '<strong>True:</strong>';
    rightLabel.style.marginTop = '10px';
    childrenDiv.appendChild(rightLabel);
    childrenDiv.appendChild(renderTreeNode(tree, tree.children_right[nodeIdx], featureMap, depth + 1));

    nodeDiv.appendChild(childrenDiv);
    return nodeDiv;
}

// Main initialization
async function init() {
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const treeContainer = document.getElementById('treeContainer');

    try {
        // Get URL params
        const params = getUrlParams();
        if (!params.label) {
            throw new Error('Missing URL parameter. Expected: ?label=module-component');
        }

        // Load tree data
        loadingDiv.textContent = 'Loading tree data...';
        const treesData = await loadTreeData();

        // Find the tree for this component
        loadingDiv.textContent = 'Finding component tree...';
        const { layerData, tree } = findComponentTree(treesData, params.module, params.component);

        // Update page title and add link to component (construct label from module:component)
        const targetLabelSpan = document.getElementById('targetLabel');
        targetLabelSpan.innerHTML = `<a href="component.html?label=${encodeURIComponent(params.label)}" style="color: #0066cc; text-decoration: none;">${params.label}</a>`;

        // Render the tree
        loadingDiv.style.display = 'none';
        const treeRoot = renderTreeNode(tree, 0, layerData.feature_map, 0);
        treeContainer.appendChild(treeRoot);

    } catch (error) {
        loadingDiv.style.display = 'none';
        errorDiv.style.display = 'block';
        errorDiv.textContent = `Error: ${error.message}`;
        console.error(error);
    }
}

// Run on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
