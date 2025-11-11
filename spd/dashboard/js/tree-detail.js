// tree-detail.js - Decision tree detail page logic

let treeData = null;
let targetComponentLabel = null;
let featureComponents = {};  // Map from feature label to component data
let treeViewer = null;
let columnSamples = {};  // Track which sample index is shown for each column (shared across all rows)
let allSamples = [];  // All available samples from activations data
let fullData = null;  // Store full ZANJ data for accessing activations

async function init() {
    // Get component label from URL
    const urlParams = new URLSearchParams(window.location.search);
    targetComponentLabel = urlParams.get('label');

    if (!targetComponentLabel) {
        document.getElementById('loading').textContent = 'No component label specified in URL';
        NOTIF.error('No component label specified in URL', null, null);
        return;
    }

    await loadData();
}

async function loadData() {
    try {
        // Load data via ZANJ
        const loader = new ZanjLoader(CONFIG.data.dataDir);
        const data = await loader.read();
        fullData = data;  // Store for accessing activations later


        // Load tree data from all_trees
        // Try direct access first
        console.log('Loading trees data...');
        const allTrees = await data.trees.all_trees;
        console.log('Trees loaded, count:', allTrees.length);

        // Find the tree for our target component
        treeData = allTrees.find(tree => tree.component_label === targetComponentLabel);

        if (!treeData) {
            const msg = `Tree not found for component: ${targetComponentLabel}`;
            NOTIF.error(msg, null, null);
            document.getElementById('loading').textContent = msg;
            return;
        }

        // Load component details from index_summaries
        const indexSummaries = await data.index_summaries;
        const allSummaries = await indexSummaries.summaries;

        // Create a map of all components by label for quick lookup
        const componentsByLabel = {};
        for (const component of allSummaries) {
            componentsByLabel[component.label] = component;
        }

        // Load the target component (what we're predicting)
        const targetComponent = componentsByLabel[targetComponentLabel];
        if (targetComponent) {
            featureComponents[targetComponentLabel] = targetComponent;
        }

        // Extract unique feature labels from the tree
        const featureLabels = Object.values(treeData.tree_dict.feature_labels);

        // Load components for each feature used in the tree
        for (const featureLabel of featureLabels) {
            const component = componentsByLabel[featureLabel];

            if (component) {
                featureComponents[featureLabel] = component;
            } else {
                console.warn(`Component not found: ${featureLabel}`);
            }
        }

        // Load all samples from activations data
        console.log('Loading sample data...');
        const tokenData = await fullData.activations.token_data;
        const tokens = await tokenData.tokens;
        const nSeqs = tokens.length;
        console.log(`Found ${nSeqs} sequences`);

        // Pick random sample indices
        const numSamples = Math.min(CONFIG.treePage.numSampleColumns * 10, nSeqs);
        const sampleIndices = [];
        for (let i = 0; i < numSamples; i++) {
            sampleIndices.push(Math.floor(Math.random() * nSeqs));
        }

        // Store sample info (token strings and index)
        allSamples = sampleIndices.map(idx => ({
            index: idx,
            token_strs: tokens[idx]
        }));
        console.log(`Loaded ${allSamples.length} samples`);

        // Initialize column samples (one sample index per column, shared across all rows)
        for (let i = 0; i < CONFIG.treePage.numSampleColumns; i++) {
            columnSamples[i] = i % allSamples.length;
        }

        displayTree();
        displayInfo();
        displayFeaturesTable();
        displayDebugInfo();

        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        console.error('Load error:', error);
        console.error('Stack:', error.stack);
        NOTIF.error('Failed to load tree data: ' + error.message, error, null);
    }
}

function displayInfo() {
    document.getElementById('targetLabel').textContent = targetComponentLabel;
    document.getElementById('treeAccuracy').textContent =
        (treeData.balanced_accuracy * 100).toFixed(2) + '%';
    document.getElementById('featuresCount').textContent =
        Object.keys(featureComponents).length;
}

/**
 * Convert sklearn decision tree format to DecisionTreeViewer format
 */
function convertSklearnTreeToViewerFormat() {
    const tree = treeData.tree_dict;
    const featureLabels = tree.feature_labels;

    // Get unique features and sort by feature index
    const uniqueFeatures = [...new Set(Object.keys(featureLabels).map(k => parseInt(k)))];
    uniqueFeatures.sort((a, b) => a - b);

    // Create rows - first row is for the target component
    const rows = [{
        feature: targetComponentLabel,
        nodeIds: []
    }];

    // Create rows - one per unique feature
    const featureIndexToRowIdx = {};
    uniqueFeatures.forEach((featIdx, idx) => {
        const rowIdx = idx + 1; // +1 because row 0 is the target
        featureIndexToRowIdx[featIdx] = rowIdx;
        rows.push({
            feature: featureLabels[featIdx],
            nodeIds: []
        });
    });

    // Add a row for leaf nodes
    rows.push({
        feature: 'Leaf',
        nodeIds: []
    });
    const leafRowIdx = rows.length - 1;

    // Create nodes
    const nodes = [];
    const nodeIdMap = {};  // Map from tree node index to viewer node id

    // Create the target component node
    const targetNodeId = 'target';
    const targetNode = {
        id: targetNodeId,
        rowIdx: 0,
        x: CONFIG.treePage.dagWidth / 2,
        y: CONFIG.treePage.rowHeight / 2 - 8,
        edges: [{
            target: 'n0',  // Depends on root node of decision tree
            type: 'depends',
            freq: 1.0
        }],
        treeNodeIdx: -1,
        isLeaf: false,
        isRoot: false,
        isTarget: true,
        threshold: null,
        nSamples: tree.n_node_samples[0]
    };
    nodes.push(targetNode);
    rows[0].nodeIds.push(targetNodeId);

    // Create decision tree nodes
    for (let i = 0; i < tree.feature.length; i++) {
        const nodeId = `n${i}`;
        nodeIdMap[i] = nodeId;

        const featIdx = tree.feature[i];
        const isLeaf = featIdx === -2;
        const isRoot = i === 0;

        // Assign row based on feature (add 1 because row 0 is target)
        const rowIdx = isLeaf ? leafRowIdx : featureIndexToRowIdx[featIdx];

        // Calculate initial position
        const x = Math.random() * (CONFIG.treePage.dagWidth - 30) + 10;
        const y = rowIdx * CONFIG.treePage.rowHeight + CONFIG.treePage.rowHeight / 2 - 8;

        const node = {
            id: nodeId,
            rowIdx: rowIdx,
            x: x,
            y: y,
            edges: [],
            treeNodeIdx: i,
            isLeaf: isLeaf,
            isRoot: isRoot,
            isTarget: false,
            threshold: tree.threshold[i],
            nSamples: tree.n_node_samples[i]
        };

        nodes.push(node);
        rows[rowIdx].nodeIds.push(nodeId);
    }

    // Create edges for decision tree nodes
    for (let i = 0; i < tree.feature.length; i++) {
        const leftChild = tree.children_left[i];
        const rightChild = tree.children_right[i];
        const nodeIdx = nodes.findIndex(n => n.id === `n${i}`);

        if (leftChild !== -1) {
            const targetId = nodeIdMap[leftChild];
            nodes[nodeIdx].edges.push({
                target: targetId,
                type: 'false',
                freq: tree.n_node_samples[leftChild] / tree.n_node_samples[i]
            });
        }

        if (rightChild !== -1) {
            const targetId = nodeIdMap[rightChild];
            nodes[nodeIdx].edges.push({
                target: targetId,
                type: 'true',
                freq: tree.n_node_samples[rightChild] / tree.n_node_samples[i]
            });
        }
    }

    return { nodes, rows };
}

function displayTree() {
    const viewerData = convertSklearnTreeToViewerFormat();

    treeViewer = new DecisionTreeViewer('treeVisualization', viewerData, {
        dagWidth: CONFIG.treePage.dagWidth,
        rowHeight: CONFIG.treePage.rowHeight
    });

    // Set up hover handlers for synchronization
    setupTreeTableSync();
}

/**
 * Set up synchronized highlighting between tree and table
 */
function setupTreeTableSync() {
    // This will be enhanced after the table is created
    // For now, we'll add event listeners to tree nodes

    // We need to modify the DecisionTreeViewer to expose node hover events
    // Or we can add listeners directly to the SVG elements
    const treeContainer = document.getElementById('treeVisualization');

    // Add listeners to all tree nodes
    treeContainer.addEventListener('mouseover', (e) => {
        if (e.target.closest('.dt-node')) {
            const nodeElement = e.target.closest('.dt-node');
            const nodeId = nodeElement.id;

            // Find the node data
            const node = treeViewer.allNodes.find(n => n.id === nodeId);
            if (node && !node.isLeaf) {
                // Highlight the corresponding table row
                const featureLabel = treeViewer.rows[node.rowIdx].feature;
                highlightTableRow(featureLabel, true);
            }
        }
    });

    treeContainer.addEventListener('mouseout', (e) => {
        if (e.target.closest('.dt-node')) {
            const nodeElement = e.target.closest('.dt-node');
            const nodeId = nodeElement.id;

            const node = treeViewer.allNodes.find(n => n.id === nodeId);
            if (node && !node.isLeaf) {
                const featureLabel = treeViewer.rows[node.rowIdx].feature;
                highlightTableRow(featureLabel, false);
            }
        }
    });
}

function highlightTableRow(featureLabel, highlight) {
    const rows = document.querySelectorAll('.feature-row');
    rows.forEach(row => {
        if (row.dataset.featureLabel === featureLabel) {
            if (highlight) {
                row.classList.add('highlighted');
            } else {
                row.classList.remove('highlighted');
            }
        }
    });
}

function highlightTreeNodes(featureLabel, highlight) {
    // Find all nodes that use this feature
    const rowIdx = treeViewer.rows.findIndex(r => r.feature === featureLabel);
    if (rowIdx === -1) return;

    const nodeIds = treeViewer.rows[rowIdx].nodeIds;
    nodeIds.forEach(nodeId => {
        const nodeEl = document.getElementById(nodeId);
        if (nodeEl) {
            if (highlight) {
                nodeEl.classList.add('highlighted');
            } else {
                nodeEl.classList.remove('highlighted');
            }
        }
    });
}

async function displayFeaturesTable() {
    const tableContainer = document.getElementById('featuresTable');

    // Prepare table data
    const tableData = [];

    // First, add the target component (what we're predicting)
    const targetComponent = featureComponents[targetComponentLabel];
    if (targetComponent) {
        const rowData = {
            featureLabel: targetComponentLabel,
            displayLabel: `[Target] ${targetComponentLabel}`,
            component: targetComponent,
            isTarget: true
        };

        // Add sample data for each column
        for (let i = 0; i < CONFIG.treePage.numSampleColumns; i++) {
            rowData[`sample_${i}`] = {
                sampleIdx: columnSamples[i],
                component: targetComponent
            };
        }

        tableData.push(rowData);
    }

    // Add all feature components used in the tree
    // Sort by feature index for consistent ordering
    const tree = treeData.tree_dict;
    const sortedFeatures = Object.entries(tree.feature_labels)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))  // Sort by feature index
        .map(([_featIdx, label]) => label);

    for (const featureLabel of sortedFeatures) {
        const component = featureComponents[featureLabel];
        if (!component) {
            console.warn(`Component not found: ${featureLabel}`);
            continue;
        }

        const rowData = {
            featureLabel: featureLabel,
            displayLabel: featureLabel,
            component: component,
            isRoot: false
        };

        // Add sample data for each column
        for (let i = 0; i < CONFIG.treePage.numSampleColumns; i++) {
            rowData[`sample_${i}`] = {
                sampleIdx: columnSamples[i],
                component: component
            };
        }

        tableData.push(rowData);
    }

    // Define columns
    const columns = [
        {
            key: 'displayLabel',
            label: 'Feature',
            type: 'string',
            width: '200px',
            renderer: (value) => {
                const span = document.createElement('span');
                span.style.fontFamily = 'monospace';
                span.style.fontSize = '11px';
                span.textContent = value;
                return span;
            }
        }
    ];

    // Add sample columns
    for (let i = 0; i < CONFIG.treePage.numSampleColumns; i++) {
        columns.push({
            key: `sample_${i}`,
            label: `Sample ${i + 1}`,
            columnIdx: i,  // Store column index for re-roll button
            type: 'custom',
            width: '300px',
            filterable: false,
            renderer: async (value, row) => {
                // Call async createSampleCell with featureLabel and global sampleIdx
                return await createSampleCell(row.featureLabel, value.sampleIdx);
            }
        });
    }

    // Create table (using basic HTML table for now, since DataTable might not handle custom renderers well)
    const table = document.createElement('table');
    table.className = 'features-table';
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';

    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');

        // For sample columns, add re-roll button in header
        if (col.columnIdx !== undefined) {
            const headerContainer = document.createElement('div');
            headerContainer.style.display = 'flex';
            headerContainer.style.justifyContent = 'space-between';
            headerContainer.style.alignItems = 'center';

            const label = document.createElement('span');
            label.textContent = col.label;

            const rerollBtn = document.createElement('button');
            rerollBtn.className = 'reroll-btn';
            rerollBtn.textContent = '↻';
            rerollBtn.title = 'Re-roll sample for this column';
            rerollBtn.onclick = () => rerollSample(col.columnIdx);

            headerContainer.appendChild(label);
            headerContainer.appendChild(rerollBtn);
            th.appendChild(headerContainer);
        } else {
            th.textContent = col.label;
        }

        th.style.padding = '10px';
        th.style.borderBottom = '2px solid #e5e7eb';
        th.style.textAlign = 'left';
        th.style.backgroundColor = '#f8f9fa';
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement('tbody');
    for (const rowData of tableData) {
        const tr = document.createElement('tr');
        tr.className = 'feature-row';
        tr.dataset.featureLabel = rowData.featureLabel;
        tr.style.borderBottom = '1px solid #e5e7eb';

        // Add hover handlers for synchronization
        tr.addEventListener('mouseenter', () => {
            highlightTreeNodes(rowData.featureLabel, true);
        });
        tr.addEventListener('mouseleave', () => {
            highlightTreeNodes(rowData.featureLabel, false);
        });

        columns.forEach(async (col) => {
            const td = document.createElement('td');
            td.style.padding = '10px';
            td.style.verticalAlign = 'top';

            const value = rowData[col.key];

            if (col.renderer) {
                // Handle async renderer for sample cells
                const rendered = await col.renderer(value, rowData);

                if (typeof rendered === 'string') {
                    td.innerHTML = rendered;
                } else if (rendered instanceof HTMLElement) {
                    td.appendChild(rendered);
                } else if (rendered instanceof Promise) {
                    // If still a promise, wait for it
                    rendered.then(r => {
                        if (r instanceof HTMLElement) {
                            td.appendChild(r);
                        } else {
                            td.textContent = String(r);
                        }
                    });
                } else {
                    td.textContent = rendered;
                }
            } else {
                td.textContent = value;
            }

            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    tableContainer.appendChild(table);
}

/**
 * Create a sample cell showing activations for a specific feature on a shared sample.
 * @param {string} featureLabel - Label of the feature component (e.g., "model.layers.0.mlp:5")
 * @param {number} sampleIdx - Index into allSamples array
 * @returns {HTMLElement} Container with token visualization
 */
async function createSampleCell(featureLabel, sampleIdx) {
    const container = document.createElement('div');
    container.className = 'sample-cell';

    if (sampleIdx >= allSamples.length || sampleIdx < 0) {
        const noData = document.createElement('span');
        noData.textContent = 'Invalid sample index';
        noData.style.color = '#999';
        noData.style.fontSize = '11px';
        container.appendChild(noData);
        return container;
    }

    const globalSample = allSamples[sampleIdx];

    // Load activations for THIS feature on THIS sample
    // We need to load enough samples to get to the one we want
    const samples = await loadComponentSamples(fullData, featureLabel, globalSample.index + 1);

    if (!samples || samples.length === 0 || globalSample.index >= samples.length) {
        const noData = document.createElement('span');
        noData.textContent = 'No activations';
        noData.style.color = '#999';
        noData.style.fontSize = '11px';
        container.appendChild(noData);
        return container;
    }

    const featureSample = samples[globalSample.index];

    // Create token visualization with this feature's activations
    const tokenViz = createTokenVisualization(featureSample.token_strs, featureSample.activations);
    container.appendChild(tokenViz);

    return container;
}

/**
 * Re-roll (change) the sample displayed in a specific column.
 * This affects ALL rows in that column (they all show the same sample).
 * @param {number} columnIdx - Index of the column to re-roll
 */
async function rerollSample(columnIdx) {
    if (allSamples.length === 0) return;

    // Pick a new random sample for this column
    columnSamples[columnIdx] = Math.floor(Math.random() * allSamples.length);

    // Redisplay the table
    document.getElementById('featuresTable').innerHTML = '';
    await displayFeaturesTable();
}

function displayDebugInfo() {
    const debugBox = document.getElementById('debugBox');
    const treeJsonPre = document.getElementById('treeJson');

    if (debugBox && treeJsonPre && treeData) {
        // Show the debug box
        debugBox.style.display = 'block';

        // Format and display the tree JSON
        const treeJson = {
            component_label: treeData.component_label,
            balanced_accuracy: treeData.balanced_accuracy,
            tree_dict: treeData.tree_dict
        };

        treeJsonPre.textContent = JSON.stringify(treeJson, null, 2);
    }
}

// Global function for minimize crossings button
function minimizeCrossings() {
    if (treeViewer) {
        treeViewer.minimizeCrossings();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await initConfig();
    init();
});
