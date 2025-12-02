/**
 * Canvas-based graph renderer for attribution visualization.
 *
 * Replaces SVG rendering with Canvas for better performance on large graphs (1k+ edges).
 *
 * Usage:
 *   const canvas = document.getElementById('graph-canvas');
 *   const renderer = new GraphCanvasRenderer(canvas);
 *
 *   // Set data (call whenever data changes)
 *   renderer.setData({ nodes, edges, seqLabels, layerLabels });
 *
 *   // Wire up callbacks for Alpine integration
 *   renderer.onNodeHover = (node) => { ... };
 *   renderer.onNodeClick = (node) => { ... };
 *   renderer.onEdgeHover = (edge) => { ... };
 *
 *   // Update visual state
 *   renderer.setHighlightedNodes(new Set(['h.0.attn.q_proj:0:5']));
 *   renderer.setHoveredNode('h.0.attn.q_proj:0:5');
 */

class GraphCanvasRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Data
        this.nodes = new Map();     // nodeKey -> {x, y, layer, cIdx, seqIdx, color, opacity}
        this.edges = [];            // [{srcKey, tgtKey, val, x1, y1, x2, y2, color, opacity}]
        this.seqLabels = [];        // [{x, text}]
        this.layerLabels = [];      // [{y, text}]

        // Visual state
        this.highlightedNodes = new Set();  // Node keys to highlight
        this.hoveredNodeKey = null;
        this.hoveredEdge = null;
        this.selectedNodeKey = null;

        // View transform (pan/zoom)
        this.transform = { x: 0, y: 0, scale: 1 };
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };

        // Layout constants
        this.nodeSize = 8;
        this.hitPadding = 4;  // Extra pixels for easier hit detection

        // Callbacks (set by consumer)
        this.onNodeHover = null;    // (node: {key, layer, cIdx, seqIdx} | null) => void
        this.onNodeClick = null;    // (node: {key, layer, cIdx, seqIdx}) => void
        this.onEdgeHover = null;    // (edge: {srcKey, tgtKey, val} | null) => void
        this.onTransformChange = null;  // (transform: {x, y, scale}) => void

        // Spatial index for fast hit testing
        this.nodeIndex = null;  // Will be a simple grid index

        // Setup
        this._setupEventListeners();
        this._setupResizeObserver();

        // Render state
        this._pendingRender = false;
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * Set graph data. Call this when edges/nodes change.
     */
    setData({ nodes, edges, seqLabels = [], layerLabels = [] }) {
        this.nodes = nodes instanceof Map ? nodes : new Map(Object.entries(nodes));
        this.edges = edges;
        this.seqLabels = seqLabels;
        this.layerLabels = layerLabels;

        this._buildSpatialIndex();
        this._scheduleRender();
    }

    /**
     * Set which nodes should be highlighted (e.g., pinned nodes).
     */
    setHighlightedNodes(nodeKeys) {
        this.highlightedNodes = nodeKeys instanceof Set ? nodeKeys : new Set(nodeKeys);
        this._scheduleRender();
    }

    /**
     * Set the currently hovered node (for external control).
     */
    setHoveredNode(nodeKey) {
        if (this.hoveredNodeKey !== nodeKey) {
            this.hoveredNodeKey = nodeKey;
            this._scheduleRender();
        }
    }

    /**
     * Set the selected/clicked node.
     */
    setSelectedNode(nodeKey) {
        this.selectedNodeKey = nodeKey;
        this._scheduleRender();
    }

    /**
     * Set view transform (pan/zoom).
     */
    setTransform(transform) {
        this.transform = { ...transform };
        this._scheduleRender();
    }

    /**
     * Reset view to fit all content.
     */
    resetView() {
        this.transform = { x: 0, y: 0, scale: 1 };
        this._scheduleRender();
        this.onTransformChange?.(this.transform);
    }

    /**
     * Force an immediate render.
     */
    render() {
        this._render();
    }

    /**
     * Clean up event listeners.
     */
    destroy() {
        this.canvas.removeEventListener('mousemove', this._onMouseMove);
        this.canvas.removeEventListener('click', this._onClick);
        this.canvas.removeEventListener('mousedown', this._onMouseDown);
        this.canvas.removeEventListener('mouseup', this._onMouseUp);
        this.canvas.removeEventListener('mouseleave', this._onMouseLeave);
        this.canvas.removeEventListener('wheel', this._onWheel);
        this._resizeObserver?.disconnect();
    }

    // =========================================================================
    // Rendering
    // =========================================================================

    _scheduleRender() {
        if (this._pendingRender) return;
        this._pendingRender = true;
        requestAnimationFrame(() => {
            this._pendingRender = false;
            this._render();
        });
    }

    _render() {
        const ctx = this.ctx;
        const { width, height } = this.canvas;

        // Clear
        ctx.clearRect(0, 0, width, height);

        // Apply transform
        ctx.save();
        ctx.translate(this.transform.x, this.transform.y);
        ctx.scale(this.transform.scale, this.transform.scale);

        // Draw layers back-to-front
        this._drawEdges(ctx);
        this._drawNodes(ctx);
        this._drawLabels(ctx);

        ctx.restore();
    }

    _drawEdges(ctx) {
        // Separate edges into highlighted and regular for draw order
        const regular = [];
        const highlighted = [];

        for (const edge of this.edges) {
            const isHighlighted = this._isEdgeHighlighted(edge);
            if (isHighlighted) {
                highlighted.push(edge);
            } else {
                regular.push(edge);
            }
        }

        // Draw regular edges first (back)
        for (const edge of regular) {
            this._drawEdge(ctx, edge, false);
        }

        // Draw highlighted edges on top
        for (const edge of highlighted) {
            this._drawEdge(ctx, edge, true);
        }
    }

    _drawEdge(ctx, edge, highlighted) {
        const { x1, y1, x2, y2, val, color, opacity } = edge;

        ctx.beginPath();
        ctx.moveTo(x1, y1);

        // Bezier curve - same logic as SVG version
        const dy = Math.abs(y2 - y1);
        const curveOffset = Math.max(20, dy * 0.4);
        ctx.bezierCurveTo(
            x1, y1 - curveOffset,
            x2, y2 + curveOffset,
            x2, y2
        );

        // Styling
        const baseColor = color || (val > 0 ? '#2196f3' : '#f44336');
        ctx.strokeStyle = baseColor;

        if (highlighted) {
            ctx.lineWidth = 2.5 / this.transform.scale;
            ctx.globalAlpha = 1;
        } else {
            const absVal = Math.abs(val);
            ctx.lineWidth = Math.max(0.5, Math.min(3, absVal * 5)) / this.transform.scale;
            ctx.globalAlpha = opacity ?? Math.max(0.1, Math.min(0.8, absVal));
        }

        ctx.stroke();
        ctx.globalAlpha = 1;
    }

    _isEdgeHighlighted(edge) {
        if (edge === this.hoveredEdge) return true;
        if (this.hoveredNodeKey) {
            return edge.srcKey === this.hoveredNodeKey || edge.tgtKey === this.hoveredNodeKey;
        }
        if (this.highlightedNodes.size > 0) {
            // Check if edge connects to any highlighted node (by component, ignoring seq position)
            for (const key of this.highlightedNodes) {
                const [layer, , cIdx] = this._parseNodeKey(key);
                const [srcLayer, , srcCIdx] = this._parseNodeKey(edge.srcKey);
                const [tgtLayer, , tgtCIdx] = this._parseNodeKey(edge.tgtKey);
                if ((srcLayer === layer && srcCIdx === cIdx) ||
                    (tgtLayer === layer && tgtCIdx === cIdx)) {
                    return true;
                }
            }
        }
        return false;
    }

    _drawNodes(ctx) {
        const size = this.nodeSize;
        const halfSize = size / 2;

        for (const [key, node] of this.nodes) {
            const isHovered = key === this.hoveredNodeKey;
            const isSelected = key === this.selectedNodeKey;
            const isHighlighted = this._isNodeHighlighted(key);

            const { x, y, color, opacity } = node;

            // Background/border for highlighted state
            if (isHovered || isSelected || isHighlighted) {
                ctx.strokeStyle = isHovered ? '#000' : '#333';
                ctx.lineWidth = (isHovered ? 2.5 : 2) / this.transform.scale;
                ctx.strokeRect(x - halfSize - 1, y - halfSize - 1, size + 2, size + 2);
            }

            // Fill
            ctx.fillStyle = color || '#666';
            ctx.globalAlpha = (isHovered || isHighlighted) ? 1 : (opacity ?? 0.7);
            ctx.fillRect(x - halfSize, y - halfSize, size, size);
            ctx.globalAlpha = 1;
        }
    }

    _isNodeHighlighted(nodeKey) {
        if (this.highlightedNodes.has(nodeKey)) return true;

        // Check if this node matches any highlighted node by layer:cIdx (ignoring seq)
        const [layer, , cIdx] = this._parseNodeKey(nodeKey);
        for (const key of this.highlightedNodes) {
            const [hLayer, , hCIdx] = this._parseNodeKey(key);
            if (layer === hLayer && cIdx === hCIdx) return true;
        }

        return false;
    }

    _drawLabels(ctx) {
        // Sequence labels (top)
        ctx.fillStyle = '#333';
        ctx.font = '11px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        for (const { x, text } of this.seqLabels) {
            ctx.fillText(text, x, -5);
        }

        // Layer labels (left) - rotate text
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        for (const { y, text } of this.layerLabels) {
            ctx.fillText(text, -10, y);
        }
    }

    // =========================================================================
    // Hit Testing
    // =========================================================================

    _buildSpatialIndex() {
        // Simple grid-based spatial index for nodes
        const cellSize = 50;
        this.nodeIndex = new Map();

        for (const [key, node] of this.nodes) {
            const cellX = Math.floor(node.x / cellSize);
            const cellY = Math.floor(node.y / cellSize);
            const cellKey = `${cellX},${cellY}`;

            if (!this.nodeIndex.has(cellKey)) {
                this.nodeIndex.set(cellKey, []);
            }
            this.nodeIndex.get(cellKey).push({ key, ...node });
        }
    }

    _hitTestNode(x, y) {
        const halfSize = this.nodeSize / 2 + this.hitPadding;
        const cellSize = 50;

        // Check nearby cells
        const cellX = Math.floor(x / cellSize);
        const cellY = Math.floor(y / cellSize);

        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                const cellKey = `${cellX + dx},${cellY + dy}`;
                const nodes = this.nodeIndex?.get(cellKey);
                if (!nodes) continue;

                for (const node of nodes) {
                    if (x >= node.x - halfSize && x <= node.x + halfSize &&
                        y >= node.y - halfSize && y <= node.y + halfSize) {
                        return node;
                    }
                }
            }
        }

        return null;
    }

    _hitTestEdge(x, y) {
        const threshold = 8 / this.transform.scale;

        for (const edge of this.edges) {
            // Quick bounding box check
            const minX = Math.min(edge.x1, edge.x2) - threshold;
            const maxX = Math.max(edge.x1, edge.x2) + threshold;
            const minY = Math.min(edge.y1, edge.y2) - threshold;
            const maxY = Math.max(edge.y1, edge.y2) + threshold;

            if (x < minX || x > maxX || y < minY || y > maxY) continue;

            // Approximate distance to bezier by sampling points
            const dist = this._distanceToBezier(x, y, edge);
            if (dist < threshold) {
                return edge;
            }
        }

        return null;
    }

    _distanceToBezier(px, py, edge) {
        const { x1, y1, x2, y2 } = edge;
        const dy = Math.abs(y2 - y1);
        const curveOffset = Math.max(20, dy * 0.4);

        // Control points for bezier
        const cp1x = x1, cp1y = y1 - curveOffset;
        const cp2x = x2, cp2y = y2 + curveOffset;

        // Sample bezier at several points and find minimum distance
        let minDist = Infinity;
        const samples = 20;

        for (let i = 0; i <= samples; i++) {
            const t = i / samples;
            const bx = this._bezierPoint(x1, cp1x, cp2x, x2, t);
            const by = this._bezierPoint(y1, cp1y, cp2y, y2, t);
            const dist = Math.hypot(px - bx, py - by);
            minDist = Math.min(minDist, dist);
        }

        return minDist;
    }

    _bezierPoint(p0, p1, p2, p3, t) {
        const mt = 1 - t;
        return mt * mt * mt * p0 +
               3 * mt * mt * t * p1 +
               3 * mt * t * t * p2 +
               t * t * t * p3;
    }

    // =========================================================================
    // Event Handling
    // =========================================================================

    _setupEventListeners() {
        // Bind methods to preserve 'this'
        this._onMouseMove = this._handleMouseMove.bind(this);
        this._onClick = this._handleClick.bind(this);
        this._onMouseDown = this._handleMouseDown.bind(this);
        this._onMouseUp = this._handleMouseUp.bind(this);
        this._onMouseLeave = this._handleMouseLeave.bind(this);
        this._onWheel = this._handleWheel.bind(this);

        this.canvas.addEventListener('mousemove', this._onMouseMove);
        this.canvas.addEventListener('click', this._onClick);
        this.canvas.addEventListener('mousedown', this._onMouseDown);
        this.canvas.addEventListener('mouseup', this._onMouseUp);
        this.canvas.addEventListener('mouseleave', this._onMouseLeave);
        this.canvas.addEventListener('wheel', this._onWheel, { passive: false });
    }

    _setupResizeObserver() {
        this._resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                const dpr = window.devicePixelRatio || 1;
                this.canvas.width = width * dpr;
                this.canvas.height = height * dpr;
                this.ctx.scale(dpr, dpr);
                this._scheduleRender();
            }
        });
        this._resizeObserver.observe(this.canvas);
    }

    _screenToWorld(screenX, screenY) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (screenX - rect.left - this.transform.x) / this.transform.scale;
        const y = (screenY - rect.top - this.transform.y) / this.transform.scale;
        return { x, y };
    }

    _handleMouseMove(e) {
        if (this.isDragging) {
            const dx = e.clientX - this.dragStart.x;
            const dy = e.clientY - this.dragStart.y;
            this.transform.x += dx;
            this.transform.y += dy;
            this.dragStart = { x: e.clientX, y: e.clientY };
            this._scheduleRender();
            this.onTransformChange?.(this.transform);
            return;
        }

        const { x, y } = this._screenToWorld(e.clientX, e.clientY);

        // Check nodes first
        const hitNode = this._hitTestNode(x, y);
        if (hitNode) {
            if (this.hoveredNodeKey !== hitNode.key) {
                this.hoveredNodeKey = hitNode.key;
                this.hoveredEdge = null;
                this.canvas.style.cursor = 'pointer';
                this.onNodeHover?.(hitNode);
                this._scheduleRender();
            }
            return;
        }

        // Then check edges
        const hitEdge = this._hitTestEdge(x, y);
        if (hitEdge) {
            if (this.hoveredEdge !== hitEdge) {
                this.hoveredEdge = hitEdge;
                this.hoveredNodeKey = null;
                this.canvas.style.cursor = 'pointer';
                this.onNodeHover?.(null);
                this.onEdgeHover?.(hitEdge);
                this._scheduleRender();
            }
            return;
        }

        // Nothing hovered
        if (this.hoveredNodeKey || this.hoveredEdge) {
            this.hoveredNodeKey = null;
            this.hoveredEdge = null;
            this.canvas.style.cursor = 'grab';
            this.onNodeHover?.(null);
            this.onEdgeHover?.(null);
            this._scheduleRender();
        }
    }

    _handleClick(e) {
        if (this.hoveredNodeKey) {
            const node = this.nodes.get(this.hoveredNodeKey);
            if (node) {
                this.onNodeClick?.({ key: this.hoveredNodeKey, ...node });
            }
        }
    }

    _handleMouseDown(e) {
        if (e.button === 0) {  // Left click
            this.isDragging = true;
            this.dragStart = { x: e.clientX, y: e.clientY };
            this.canvas.style.cursor = 'grabbing';
        }
    }

    _handleMouseUp(e) {
        this.isDragging = false;
        this.canvas.style.cursor = this.hoveredNodeKey ? 'pointer' : 'grab';
    }

    _handleMouseLeave(e) {
        this.isDragging = false;
        if (this.hoveredNodeKey || this.hoveredEdge) {
            this.hoveredNodeKey = null;
            this.hoveredEdge = null;
            this.onNodeHover?.(null);
            this.onEdgeHover?.(null);
            this._scheduleRender();
        }
    }

    _handleWheel(e) {
        e.preventDefault();

        const { x, y } = this._screenToWorld(e.clientX, e.clientY);

        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(0.1, Math.min(5, this.transform.scale * zoomFactor));

        // Zoom towards cursor position
        this.transform.x = e.clientX - this.canvas.getBoundingClientRect().left - x * newScale;
        this.transform.y = e.clientY - this.canvas.getBoundingClientRect().top - y * newScale;
        this.transform.scale = newScale;

        this._scheduleRender();
        this.onTransformChange?.(this.transform);
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    _parseNodeKey(key) {
        // Format: "layer:seqIdx:cIdx" e.g. "h.0.attn.q_proj:2:5"
        const parts = key.split(':');
        const layer = parts.slice(0, -2).join(':');
        const seqIdx = parseInt(parts[parts.length - 2], 10);
        const cIdx = parseInt(parts[parts.length - 1], 10);
        return [layer, seqIdx, cIdx];
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GraphCanvasRenderer;
}
