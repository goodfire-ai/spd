/**
 * DecisionTreeViewer - A library for visualizing decision trees with interactive node manipulation
 * 
 * Usage:
 *   const viewer = new DecisionTreeViewer('container-id', {
 *     nodes: [...],
 *     rows: [...]
 *   }, {
 *     dagWidth: 500,
 *     rowHeight: 40
 *   });
 */

class DecisionTreeViewer {
    constructor(containerId, data, config = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id "${containerId}" not found`);
        }

        // Config with defaults
        this.config = {
            dagWidth: config.dagWidth || 500,
            rowHeight: config.rowHeight || 40,
            ...config
        };

        // Data
        this.allNodes = data.nodes || [];
        this.rows = data.rows || [];

        // State
        this.draggedNode = null;
        this.dragOffset = 0;

        // Initialize
        this.setupDOM();
        this.attachEventListeners();
        this.render();
    }

    setupDOM() {
        this.container.innerHTML = `
            <div class="dt-viewer">
                <style>
                    .dt-viewer { font-family: 'Courier New', monospace; }
                    .dt-dag-container { background: white; }
                    .dt-node { cursor: move; }
                    .dt-node.highlighted rect { fill: #fbbf24 !important; }
                    .dt-edge.highlighted { filter: drop-shadow(0 0 4px #fbbf24); }
                    .dt-connector.highlighted { stroke: #fbbf24 !important; stroke-width: 3 !important; }
                </style>
                <div class="dt-dag-container"></div>
            </div>
        `;

        this.dagContainer = this.container.querySelector('.dt-dag-container');
    }

    attachEventListeners() {
        document.addEventListener('mousemove', (e) => this.drag(e));
        document.addEventListener('mouseup', () => this.endDrag());
    }

    getNode(id) {
        return this.allNodes.find(n => n.id === id);
    }

    getNodeType(node) {
        const hasIncoming = this.allNodes.some(n => n.edges.some(e => e.target === node.id));
        const hasTrue = node.edges.some(e => e.type === 'true');
        const hasFalse = node.edges.some(e => e.type === 'false');
        const hasDepends = node.edges.some(e => e.type === 'depends');
        
        if (!hasIncoming && node.edges.length === 0) return 'isolated';
        if (hasIncoming && node.edges.length === 0) return 'leaf';
        if (hasTrue || hasFalse) return 'branch';
        if (hasDepends) return 'root';
        return 'isolated';
    }

    highlightPath(nodeId, highlight) {
        const visited = new Set();
        const toHighlight = new Set([nodeId]);

        // BFS to find all ancestors
        const queue = [nodeId];
        while (queue.length > 0) {
            const curr = queue.shift();
            if (visited.has(curr)) continue;
            visited.add(curr);

            this.allNodes.forEach(sourceNode => {
                sourceNode.edges.forEach(edge => {
                    if (edge.target === curr) {
                        toHighlight.add(sourceNode.id);
                        toHighlight.add(`edge-${sourceNode.id}-${curr}`);
                        queue.push(sourceNode.id);
                    }
                });
            });
        }

        // Highlight outgoing depends edges
        const node = this.getNode(nodeId);
        if (node) {
            node.edges.forEach(edge => {
                if (edge.type === 'depends') {
                    toHighlight.add(edge.target);
                    toHighlight.add(`edge-${nodeId}-${edge.target}`);
                }
            });
        }

        // Clear all highlights
        this.container.querySelectorAll('.dt-node, .dt-edge, .dt-connector').forEach(el => {
            el.classList.remove('highlighted');
        });

        // Apply highlights
        if (highlight) {
            toHighlight.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.add('highlighted');
            });

            const conn = document.getElementById(`conn-${node.rowIdx}`);
            if (conn) conn.classList.add('highlighted');
        }
    }

    startDrag(e, nodeId) {
        this.draggedNode = nodeId;
        const node = this.getNode(nodeId);
        this.dragOffset = e.clientX - node.x;
    }

    drag(e) {
        if (!this.draggedNode) return;
        const node = this.getNode(this.draggedNode);
        node.x = Math.max(0, Math.min(e.clientX - this.dragOffset, this.config.dagWidth - 30));
        
        const nodeEl = document.getElementById(this.draggedNode);
        const rect = nodeEl.querySelector('rect');
        const text = nodeEl.querySelector('text');
        rect.setAttribute('x', node.x);
        text.setAttribute('x', node.x + 10.5);
        
        this.updateAllEdges();
        this.updateConnectors();
    }

    endDrag() {
        this.draggedNode = null;
    }

    updateAllEdges() {
        this.allNodes.forEach(sourceNode => {
            sourceNode.edges.forEach(e => {
                const targetNode = this.getNode(e.target);
                const edgeEl = document.getElementById(`edge-${sourceNode.id}-${e.target}`);
                if (!edgeEl || !targetNode) return;

                const sx = sourceNode.x + 10.5;
                const sy = sourceNode.y + 16;
                const tx = targetNode.x + 10.5;
                const ty = targetNode.y;
                const midY = (sy + ty) / 2;
                const d = `M ${sx} ${sy} C ${sx} ${midY}, ${tx} ${midY}, ${tx} ${ty}`;
                edgeEl.setAttribute('d', d);
            });
        });
    }

    updateConnectors() {
        this.rows.forEach((row, idx) => {
            if (row.nodeIds.length > 1) {
                const n1 = this.getNode(row.nodeIds[0]);
                const n2 = this.getNode(row.nodeIds[row.nodeIds.length - 1]);
                const line = document.getElementById(`conn-${idx}`);
                if (line) {
                    line.setAttribute('x1', n1.x + 10.5);
                    line.setAttribute('x2', n2.x + 10.5);
                }
            }
        });
    }

    countCrossings() {
        const edges = [];
        this.allNodes.forEach(node => {
            node.edges.forEach(e => {
                const target = this.getNode(e.target);
                if (!target) return;
                edges.push({
                    x1: node.x + 10.5, y1: node.y + 16,
                    x2: target.x + 10.5, y2: target.y
                });
            });
        });
        
        let crossings = 0;
        for (let i = 0; i < edges.length; i++) {
            for (let j = i + 1; j < edges.length; j++) {
                const e1 = edges[i], e2 = edges[j];
                const ccw = (ax, ay, bx, by, cx, cy) => (cy - ay) * (bx - ax) > (by - ay) * (cx - ax);
                if (ccw(e1.x1, e1.y1, e2.x1, e2.y1, e2.x2, e2.y2) !== ccw(e1.x2, e1.y2, e2.x1, e2.y1, e2.x2, e2.y2) &&
                    ccw(e1.x1, e1.y1, e1.x2, e1.y2, e2.x1, e2.y1) !== ccw(e1.x1, e1.y1, e1.x2, e1.y2, e2.x2, e2.y2)) {
                    crossings++;
                }
            }
        }
        return crossings;
    }

    minimizeCrossings() {
        let temp = 50, best = this.countCrossings(), current = best;
        const bestX = new Map(this.allNodes.map(n => [n.id, n.x]));
        
        console.log(`Starting optimization: ${current} crossings`);
        
        for (let i = 0; i < 3000; i++) {
            const node = this.allNodes[Math.floor(Math.random() * this.allNodes.length)];
            const oldX = node.x;
            node.x = Math.random() * (this.config.dagWidth - 30) + 10;
            
            const newScore = this.countCrossings();
            if (newScore < current || Math.random() < Math.exp(-(newScore - current) / temp)) {
                current = newScore;
                if (newScore < best) {
                    best = newScore;
                    this.allNodes.forEach(n => bestX.set(n.id, n.x));
                    console.log(`Iteration ${i}: ${best} crossings`);
                }
            } else {
                node.x = oldX;
            }
            temp *= 0.997;
        }
        
        this.allNodes.forEach(n => n.x = bestX.get(n.id));
        console.log(`Final: ${best} crossings`);
        this.render();
    }

    render() {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        const svgHeight = this.rows.length * this.config.rowHeight;
        svg.setAttribute('width', this.config.dagWidth);
        svg.setAttribute('height', svgHeight);

        // Connectors
        this.rows.forEach((row, idx) => {
            if (row.nodeIds.length > 1) {
                const n1 = this.getNode(row.nodeIds[0]);
                const n2 = this.getNode(row.nodeIds[row.nodeIds.length - 1]);
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('id', `conn-${idx}`);
                line.classList.add('dt-connector');
                line.setAttribute('x1', n1.x + 10.5);
                line.setAttribute('y1', n1.y + 8);
                line.setAttribute('x2', n2.x + 10.5);
                line.setAttribute('y2', n2.y + 8);
                line.setAttribute('stroke', '#d1d5db');
                line.setAttribute('stroke-width', '2');
                line.setAttribute('stroke-dasharray', '3,3');
                svg.appendChild(line);
            }
        });

        // Edges
        this.allNodes.forEach(node => {
            node.edges.forEach(e => {
                const target = this.getNode(e.target);
                if (!target) return;

                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('id', `edge-${node.id}-${e.target}`);
                path.classList.add('dt-edge');

                const sx = node.x + 10.5, sy = node.y + 16;
                const tx = target.x + 10.5, ty = target.y;
                const midY = (sy + ty) / 2;
                const d = `M ${sx} ${sy} C ${sx} ${midY}, ${tx} ${midY}, ${tx} ${ty}`;
                path.setAttribute('d', d);
                path.setAttribute('fill', 'none');

                const color = e.type === 'true' ? '#22c55e' : e.type === 'false' ? '#ef4444' : '#6b7280';
                path.setAttribute('stroke', color);
                path.setAttribute('stroke-width', 0.5 + e.freq * 2.5);
                if (e.type === 'depends') path.setAttribute('stroke-dasharray', '4,2');
                svg.appendChild(path);
            });
        });

        // Nodes
        this.allNodes.forEach(node => {
            const type = this.getNodeType(node);
            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('id', node.id);
            g.classList.add('dt-node');

            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', node.x);
            rect.setAttribute('y', node.y);
            rect.setAttribute('width', '21');
            rect.setAttribute('height', '16');
            rect.setAttribute('rx', '3');
            
            const fill = type === 'branch' ? '#3b82f6' : type === 'root' ? '#8b5cf6' :
                        type === 'leaf' ? '#10b981' : '#9ca3af';
            rect.setAttribute('fill', fill);
            rect.setAttribute('stroke', '#1f2937');
            rect.setAttribute('stroke-width', type === 'branch' ? '2' : '1');
            g.appendChild(rect);

            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', node.x + 10.5);
            text.setAttribute('y', node.y + 11);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '7');
            text.setAttribute('fill', 'white');
            text.setAttribute('pointer-events', 'none');
            text.textContent = node.rowIdx;
            g.appendChild(text);

            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            title.textContent = `${this.rows[node.rowIdx].feature}\nType: ${type}`;
            g.appendChild(title);

            g.addEventListener('mouseenter', () => this.highlightPath(node.id, true));
            g.addEventListener('mouseleave', () => this.highlightPath(node.id, false));
            g.addEventListener('mousedown', (e) => this.startDrag(e, node.id));

            svg.appendChild(g);
        });

        this.dagContainer.innerHTML = '';
        this.dagContainer.style.width = this.config.dagWidth + 'px';
        this.dagContainer.appendChild(svg);
    }

    // Public API
    setData(data) {
        this.allNodes = data.nodes || this.allNodes;
        this.rows = data.rows || this.rows;
        this.render();
    }

    updateConfig(config) {
        this.config = { ...this.config, ...config };
        this.render();
    }
}