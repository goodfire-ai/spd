// tree-viewer-v2.js - Interactive decision tree viewer with D3.js and Canvas

// ===== UTILITY FUNCTIONS (reused from tree-viewer.js) =====

function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const label = params.get('label');
    if (!label) {
        return { label: null, module: null, component: null };
    }

    const lastColon = label.lastIndexOf(':');
    if (lastColon === -1) {
        throw new Error(`Invalid label format: ${label}. Expected format: module:component`);
    }

    const module = label.substring(0, lastColon);
    const component = parseInt(label.substring(lastColon + 1));

    return { label, module, component };
}

async function loadTreeData() {
    const response = await fetch('data/trees.json');
    if (!response.ok) {
        throw new Error('Failed to load trees.json');
    }
    return await response.json();
}

function findComponentTree(treesData, targetModule, targetComponent) {
    for (const layerData of treesData) {
        if (layerData.module_key === targetModule) {
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

// ===== LABEL FORMATTING =====

function formatLabel(label) {
    // Strip "model.layers." prefix
    let formatted = label.replace(/^model\.layers\./, '');

    // Strip "_proj" before the colon
    formatted = formatted.replace(/_proj:/, ':');

    return formatted;
}

// ===== TREE CONVERSION =====

function convertToD3Tree(tree, featureMap, nodeIdx = 0, maxSamples = null) {
    // Get max samples from root if not provided
    if (maxSamples === null) {
        maxSamples = tree.n_node_samples[0];
    }

    const feature = tree.feature[nodeIdx];
    const threshold = tree.threshold[nodeIdx];
    const nSamples = tree.n_node_samples[nodeIdx];
    const value = tree.value[nodeIdx];

    const node = {
        id: `node_${nodeIdx}`,
        nodeIdx: nodeIdx,
        nSamples: nSamples,
        maxSamples: maxSamples,
        children: []
    };

    // Check if leaf node
    if (feature === -2) {
        node.isLeaf = true;
        const class0 = value[0][0];
        const class1 = value[0][1];
        node.prediction = class1 > class0 ? 1 : 0;
        node.class0 = class0;
        node.class1 = class1;
        node.displayLabel = `Leaf: ${node.prediction}`;
    } else {
        node.isLeaf = false;
        const componentInfo = featureMap[feature];
        if (!componentInfo) {
            node.displayLabel = `ERROR: Invalid feature ${feature}`;
            node.fullLabel = null;
        } else {
            node.fullLabel = `${componentInfo.module_key}:${componentInfo.component_idx}`;
            node.displayLabel = formatLabel(node.fullLabel);
            node.feature = feature;
        }

        // Add children (left = false, right = true)
        const leftIdx = tree.children_left[nodeIdx];
        const rightIdx = tree.children_right[nodeIdx];

        if (leftIdx !== -1) {
            const leftChild = convertToD3Tree(tree, featureMap, leftIdx, maxSamples);
            leftChild.branch = 'false';
            node.children.push(leftChild);
        }

        if (rightIdx !== -1) {
            const rightChild = convertToD3Tree(tree, featureMap, rightIdx, maxSamples);
            rightChild.branch = 'true';
            node.children.push(rightChild);
        }
    }

    return node;
}

// ===== TREE VISUALIZATION CLASS =====

class TreeVisualization {
    constructor(canvasId, containerId) {
        this.canvas = document.getElementById(canvasId);
        this.container = document.getElementById(containerId);
        this.ctx = this.canvas.getContext('2d');
        this.tooltip = document.getElementById('nodeTooltip');

        // Visualization state
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;

        // Menu state
        this.selectedNode = null;

        // Tree data
        this.treeRoot = null;
        this.treeLayout = null;
        this.allTreesData = null; // Store for inline expansion

        // Node dimensions
        this.nodeWidth = 180;
        this.nodeHeight = 60;
        this.verticalSpacing = 80;
        this.horizontalSpacing = 20;

        this.setupCanvas();
        this.setupEventListeners();
    }

    setupCanvas() {
        // Set canvas size to match container
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    setupEventListeners() {
        // Zoom buttons
        document.getElementById('zoomIn').addEventListener('click', () => this.changeZoom(1.2));
        document.getElementById('zoomOut').addEventListener('click', () => this.changeZoom(0.8));
        document.getElementById('resetView').addEventListener('click', () => this.resetView());

        // Mouse wheel zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
            this.zoomAt(mouseX, mouseY, zoomFactor);
        });

        // Pan with mouse drag
        this.canvas.addEventListener('mousedown', (e) => {
            this.dragStartX = e.clientX;
            this.dragStartY = e.clientY;
            this.isDragging = false; // Will set to true if they actually drag
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (this.dragStartX !== null && this.dragStartY !== null) {
                const dx = e.clientX - this.dragStartX;
                const dy = e.clientY - this.dragStartY;

                // If moved more than 5 pixels, consider it a drag
                if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                    this.isDragging = true;
                    this.container.classList.add('grabbing');
                }

                if (this.isDragging) {
                    this.offsetX += dx;
                    this.offsetY += dy;
                    this.dragStartX = e.clientX;
                    this.dragStartY = e.clientY;
                    this.render();
                }
            }
        });

        this.canvas.addEventListener('mouseup', (e) => {
            const wasDragging = this.isDragging;
            this.isDragging = false;
            this.dragStartX = null;
            this.dragStartY = null;
            this.container.classList.remove('grabbing');

            // If not dragging, treat as click
            if (!wasDragging) {
                this.handleClick(e);
            }
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.dragStartX = null;
            this.dragStartY = null;
            this.container.classList.remove('grabbing');
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.setupCanvas();
            this.render();
        });
    }

    changeZoom(factor) {
        // Zoom towards center of canvas
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        this.zoomAt(centerX, centerY, factor);
    }

    zoomAt(mouseX, mouseY, factor) {
        // Get the world coordinates before zoom
        const worldX = (mouseX - this.offsetX) / this.zoom;
        const worldY = (mouseY - this.offsetY) / this.zoom;

        // Apply zoom
        const newZoom = this.zoom * factor;
        this.zoom = Math.max(0.1, Math.min(5, newZoom)); // Clamp between 0.1x and 5x

        // Adjust offset so the world point stays under the mouse
        this.offsetX = mouseX - worldX * this.zoom;
        this.offsetY = mouseY - worldY * this.zoom;

        document.getElementById('zoomLevel').textContent = `${Math.round(this.zoom * 100)}%`;
        this.render();
    }

    resetView() {
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        document.getElementById('zoomLevel').textContent = '100%';
        this.render();
    }

    setTreeData(treeRoot, allTreesData) {
        this.treeRoot = treeRoot;
        this.allTreesData = allTreesData;
        this.computeLayout();
        this.centerTree();
        this.render();
    }

    computeLayout() {
        // Use D3 tree layout
        const hierarchy = d3.hierarchy(this.treeRoot);
        const treeLayout = d3.tree()
            .nodeSize([this.nodeWidth + this.horizontalSpacing, this.nodeHeight + this.verticalSpacing]);

        this.treeLayout = treeLayout(hierarchy);
    }

    centerTree() {
        // Center the tree horizontally
        if (this.treeLayout) {
            const bounds = this.getTreeBounds();
            this.offsetX = (this.canvas.width - (bounds.maxX - bounds.minX) * this.zoom) / 2 - bounds.minX * this.zoom;
            this.offsetY = 50; // Start from top with some padding
        }
    }

    getTreeBounds() {
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        this.treeLayout.each(node => {
            minX = Math.min(minX, node.x);
            maxX = Math.max(maxX, node.x);
            minY = Math.min(minY, node.y);
            maxY = Math.max(maxY, node.y);
        });

        return { minX, maxX, minY, maxY };
    }

    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.treeLayout) return;

        this.ctx.save();
        this.ctx.translate(this.offsetX, this.offsetY);
        this.ctx.scale(this.zoom, this.zoom);

        // Draw edges first
        this.treeLayout.links().forEach(link => {
            this.drawEdge(link);
        });

        // Draw nodes
        this.treeLayout.each(node => {
            this.drawNode(node);
        });

        this.ctx.restore();
    }

    drawEdge(link) {
        const source = link.source;
        const target = link.target;

        this.ctx.beginPath();
        this.ctx.strokeStyle = '#999';
        this.ctx.lineWidth = 2;

        // Draw straight lines
        this.ctx.moveTo(source.x, source.y + this.nodeHeight / 2);
        this.ctx.lineTo(target.x, target.y - this.nodeHeight / 2);

        this.ctx.stroke();

        // Draw branch label (true/false)
        if (target.data.branch) {
            const midX = (source.x + target.x) / 2;
            const midY = (source.y + this.nodeHeight / 2 + target.y - this.nodeHeight / 2) / 2;

            this.ctx.fillStyle = target.data.branch === 'true' ? '#28a745' : '#dc3545';
            this.ctx.font = 'bold 12px monospace';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(target.data.branch, midX, midY);
        }
    }

    drawNode(node) {
        const data = node.data;
        const x = node.x;
        const y = node.y;

        // Calculate brightness based on sample ratio
        const sampleRatio = data.nSamples / data.maxSamples;
        const brightness = Math.floor(240 - (sampleRatio * 140)); // Range from 240 (light) to 100 (dark)

        // Different color for leaf nodes
        const bgColor = data.isLeaf ?
            `rgb(${brightness}, ${255}, ${brightness})` : // Green tint for leaves
            `rgb(${brightness}, ${brightness}, ${255})`; // Blue tint for split nodes

        // Draw node rectangle
        this.ctx.fillStyle = bgColor;
        this.ctx.fillRect(
            x - this.nodeWidth / 2,
            y - this.nodeHeight / 2,
            this.nodeWidth,
            this.nodeHeight
        );

        // Draw border
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(
            x - this.nodeWidth / 2,
            y - this.nodeHeight / 2,
            this.nodeWidth,
            this.nodeHeight
        );

        // Draw text
        this.ctx.fillStyle = '#000';
        this.ctx.font = '12px monospace';
        this.ctx.textAlign = 'center';

        // Split label into multiple lines if needed
        const label = data.displayLabel;
        const maxWidth = this.nodeWidth - 10;
        const lines = this.wrapText(label, maxWidth);

        const lineHeight = 14;
        const startY = y - (lines.length - 1) * lineHeight / 2;

        lines.forEach((line, i) => {
            this.ctx.fillText(line, x, startY + i * lineHeight);
        });

        // Draw sample count
        this.ctx.font = 'bold 11px monospace';
        this.ctx.fillStyle = '#000';
        this.ctx.fillText(
            `${data.nSamples} samples`,
            x,
            y + this.nodeHeight / 2 - 5
        );
    }

    wrapText(text, maxWidth) {
        const words = text.split(/(\s+|:)/);
        const lines = [];
        let currentLine = '';

        for (const word of words) {
            const testLine = currentLine + word;
            const metrics = this.ctx.measureText(testLine);

            if (metrics.width > maxWidth && currentLine !== '') {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }

        if (currentLine) {
            lines.push(currentLine);
        }

        return lines;
    }

    getNodeAt(mouseX, mouseY) {
        // Transform mouse coordinates to tree space
        const treeX = (mouseX - this.offsetX) / this.zoom;
        const treeY = (mouseY - this.offsetY) / this.zoom;

        let foundNode = null;
        this.treeLayout.each(node => {
            const left = node.x - this.nodeWidth / 2;
            const right = node.x + this.nodeWidth / 2;
            const top = node.y - this.nodeHeight / 2;
            const bottom = node.y + this.nodeHeight / 2;

            if (treeX >= left && treeX <= right && treeY >= top && treeY <= bottom) {
                foundNode = node;
            }
        });

        return foundNode;
    }

    handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const node = this.getNodeAt(mouseX, mouseY);

        // Toggle menu on click
        if (node && !node.data.isLeaf && node.data.fullLabel) {
            if (this.selectedNode === node) {
                // Clicking same node - hide menu
                this.hideMenu();
            } else {
                // Clicking different node - show menu
                this.selectedNode = node;
                this.showMenu(node, e.clientX, e.clientY);
            }
        } else {
            // Clicked elsewhere - hide menu
            this.hideMenu();
        }
    }

    showMenu(node, screenX, screenY) {
        const label = node.data.fullLabel;
        const displayLabel = node.data.displayLabel;

        this.tooltip.innerHTML = `
            <div class="tooltip-label">${displayLabel}</div>
            <div class="tooltip-info">${node.data.nSamples} samples</div>
            <a href="component.html?label=${encodeURIComponent(label)}" target="_blank">View Component</a>
            <a href="tree.html?label=${encodeURIComponent(label)}" target="_blank">View Tree (List)</a>
            <a href="tree-v2.html?label=${encodeURIComponent(label)}" target="_blank">View Tree (Graph)</a>
            <button onclick="window.treeViz.expandTreeInline('${label}')">Expand Tree Inline</button>
        `;

        this.tooltip.style.left = `${screenX + 10}px`;
        this.tooltip.style.top = `${screenY + 10}px`;
        this.tooltip.style.display = 'block';
    }

    hideMenu() {
        this.selectedNode = null;
        this.tooltip.style.display = 'none';
    }

    async expandTreeInline(label) {
        try {
            this.hideMenu();

            // Parse label
            const lastColon = label.lastIndexOf(':');
            const module = label.substring(0, lastColon);
            const component = parseInt(label.substring(lastColon + 1));

            // Find the tree for this component
            const { layerData, tree } = findComponentTree(this.allTreesData, module, component);

            // Convert to D3 tree
            const subtreeRoot = convertToD3Tree(tree, layerData.feature_map);

            // Find the node in current tree that matches this label
            this.inlineSubtree(this.treeRoot, label, subtreeRoot);

            // Recompute layout and render
            this.computeLayout();
            this.render();

        } catch (error) {
            console.error('Failed to expand tree inline:', error);
            alert(`Failed to expand tree: ${error.message}`);
        }
    }

    inlineSubtree(currentNode, targetLabel, subtreeRoot) {
        // Recursively search for the node with targetLabel
        if (currentNode.fullLabel === targetLabel) {
            // Found the target node - replace its children with subtree's children
            if (subtreeRoot.children && subtreeRoot.children.length > 0) {
                currentNode.children = subtreeRoot.children;
                currentNode.displayLabel += ' [expanded]';
                return true;
            }
        }

        // Search in children
        if (currentNode.children) {
            for (const child of currentNode.children) {
                if (this.inlineSubtree(child, targetLabel, subtreeRoot)) {
                    return true;
                }
            }
        }

        return false;
    }
}

// ===== MAIN INITIALIZATION =====

let treeViz;

async function init() {
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    try {
        // Get URL params
        const params = getUrlParams();
        if (!params.label) {
            throw new Error('Missing URL parameter. Expected: ?label=module:component');
        }

        // Load tree data
        loadingDiv.textContent = 'Loading tree data...';
        const treesData = await loadTreeData();

        // Find the tree for this component
        loadingDiv.textContent = 'Finding component tree...';
        const { layerData, tree } = findComponentTree(treesData, params.module, params.component);

        // Update page title
        const targetLabelSpan = document.getElementById('targetLabel');
        targetLabelSpan.innerHTML = `<a href="component.html?label=${encodeURIComponent(params.label)}" style="color: #0066cc; text-decoration: none;">${params.label}</a>`;

        // Convert to D3 tree structure
        loadingDiv.textContent = 'Building tree visualization...';
        const treeRoot = convertToD3Tree(tree, layerData.feature_map);

        // Create visualization
        loadingDiv.style.display = 'none';
        treeViz = new TreeVisualization('treeCanvas', 'canvasContainer');
        window.treeViz = treeViz; // Expose globally for inline expansion
        treeViz.setTreeData(treeRoot, treesData);

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
