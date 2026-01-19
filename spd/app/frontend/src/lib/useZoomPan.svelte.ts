/**
 * Shared zoom/pan state and handlers for SVG graph visualizations.
 *
 * Usage:
 *   let container: HTMLDivElement;
 *   const zoom = useZoomPan(() => container);
 *
 *   // In template:
 *   <div bind:this={container} onwheel={zoom.handleWheel} onmousemove={zoom.updatePan} ...>
 *     <svg width={width * zoom.scale + Math.max(zoom.translateX, 0)} ...>
 *       <g transform="translate({zoom.translateX}, {zoom.translateY}) scale({zoom.scale})">
 *
 * Pan-start logic differs between components, so call zoom.startPan(event)
 * from your own mousedown handler after checking your conditions.
 */

const MIN_SCALE = 0.25;
const MAX_SCALE = 4;
const ZOOM_SENSITIVITY = 0.002;

export function useZoomPan(getContainer: () => HTMLElement | null) {
    let scale = $state(1);
    let translateX = $state(0);
    let translateY = $state(0);
    let isPanning = $state(false);

    // Not reactive - only used internally
    let panStart: { x: number; y: number; tx: number; ty: number } | null = null;

    function zoomAt(px: number, py: number, factor: number) {
        const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
        if (newScale === scale) return;

        const ratio = newScale / scale;
        translateX = px - (px - translateX) * ratio;
        translateY = py - (py - translateY) * ratio;
        scale = newScale;
    }

    function handleWheel(event: WheelEvent) {
        event.preventDefault();
        const container = getContainer();
        if (!container) return;

        const rect = container.getBoundingClientRect();
        const mouseX = event.clientX - rect.left + container.scrollLeft;
        const mouseY = event.clientY - rect.top + container.scrollTop;

        zoomAt(mouseX, mouseY, 1 - event.deltaY * ZOOM_SENSITIVITY);
    }

    /** Call from mousedown handler after checking your pan-start conditions */
    function startPan(event: MouseEvent) {
        event.preventDefault();
        isPanning = true;
        panStart = { x: event.clientX, y: event.clientY, tx: translateX, ty: translateY };
    }

    function updatePan(event: MouseEvent) {
        if (!isPanning || !panStart) return;
        translateX = panStart.tx + (event.clientX - panStart.x);
        translateY = panStart.ty + (event.clientY - panStart.y);
    }

    function endPan() {
        isPanning = false;
        panStart = null;
    }

    function zoomIn() {
        const container = getContainer();
        if (!container) return;
        const x = container.clientWidth / 2 + container.scrollLeft;
        const y = container.clientHeight / 2 + container.scrollTop;
        zoomAt(x, y, 1.25);
    }

    function zoomOut() {
        const container = getContainer();
        if (!container) return;
        const x = container.clientWidth / 2 + container.scrollLeft;
        const y = container.clientHeight / 2 + container.scrollTop;
        zoomAt(x, y, 0.8);
    }

    function reset() {
        scale = 1;
        translateX = 0;
        translateY = 0;
    }

    return {
        get scale() {
            return scale;
        },
        get translateX() {
            return translateX;
        },
        get translateY() {
            return translateY;
        },
        get isPanning() {
            return isPanning;
        },
        handleWheel,
        startPan,
        updatePan,
        endPan,
        zoomIn,
        zoomOut,
        reset,
    };
}
