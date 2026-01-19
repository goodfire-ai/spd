/**
 * Utility functions for the PromptAttributionsGraph component.
 */

/** Linear interpolation between min and max. */
export function lerp(min: number, max: number, t: number): number {
    return min + (max - min) * t;
}

/** Calculate tooltip position that stays within viewport bounds. */
export function calcTooltipPos(mouseX: number, mouseY: number): { x: number; y: number } {
    const padding = 15;
    const tooltipMaxWidth = 800;
    // Tooltip has max-height: 80vh, so use that as the estimate
    const tooltipHeight = typeof window !== "undefined" ? window.innerHeight * 0.8 : 400;

    let left = mouseX + padding;
    let top = mouseY + padding;

    if (typeof window !== "undefined") {
        // If tooltip would overflow right edge, position to left of cursor
        if (left + tooltipMaxWidth > window.innerWidth) {
            left = mouseX - tooltipMaxWidth - padding;
        }
        // Clamp to left edge
        left = Math.max(padding, left);

        // Clamp to bottom of screen (don't flip, just constrain)
        if (top + tooltipHeight > window.innerHeight) {
            top = window.innerHeight - tooltipHeight - padding;
        }
    }
    return { x: Math.max(0, left), y: Math.max(0, top) };
}

/**
 * Sort component indices by importance (CI for internal nodes, probability for output nodes).
 * Returns a new sorted array (highest importance first).
 */
export function sortComponentsByImportance(
    components: number[],
    layer: string,
    seqIdx: number,
    nodeCiVals: Record<string, number>,
    outputProbs: Record<string, { prob: number }>,
): number[] {
    const isOutput = layer === "output";
    return [...components].sort((a, b) => {
        if (isOutput) {
            const keyA = `${seqIdx}:${a}`;
            const keyB = `${seqIdx}:${b}`;
            return (outputProbs[keyB]?.prob ?? 0) - (outputProbs[keyA]?.prob ?? 0);
        }
        const keyA = `${layer}:${seqIdx}:${a}`;
        const keyB = `${layer}:${seqIdx}:${b}`;
        return (nodeCiVals[keyB] ?? 0) - (nodeCiVals[keyA] ?? 0);
    });
}

/**
 * Sort component indices by cluster, then by CI within each cluster.
 * Clusters are sorted by size (biggest first), with singletons (null cluster) at the end.
 * Returns a new sorted array.
 */
export function sortComponentsByCluster(
    components: number[],
    layer: string,
    seqIdx: number,
    nodeCiVals: Record<string, number>,
    getClusterId: (layer: string, componentIdx: number) => number | null | undefined,
): number[] {
    // Group components by cluster ID
    const clusterGroups = new Map<number | null, number[]>();
    const singletons: number[] = [];

    for (const cIdx of components) {
        const clusterId = getClusterId(layer, cIdx);
        if (clusterId === undefined || clusterId === null) {
            singletons.push(cIdx);
        } else {
            const group = clusterGroups.get(clusterId);
            if (group) {
                group.push(cIdx);
            } else {
                clusterGroups.set(clusterId, [cIdx]);
            }
        }
    }

    // Sort each cluster group by CI (descending)
    const sortByCI = (a: number, b: number) => {
        const keyA = `${layer}:${seqIdx}:${a}`;
        const keyB = `${layer}:${seqIdx}:${b}`;
        if (!(keyA in nodeCiVals) || !(keyB in nodeCiVals))
            throw new Error(`Node CI value not found for key: ${keyA} or ${keyB}`);
        return nodeCiVals[keyB] - nodeCiVals[keyA];
    };

    for (const group of clusterGroups.values()) {
        group.sort(sortByCI);
    }
    singletons.sort(sortByCI);

    // Sort clusters by size (descending), then concatenate
    const sortedClusters = [...clusterGroups.entries()].sort((a, b) => b[1].length - a[1].length);

    const result: number[] = [];
    for (const [, group] of sortedClusters) {
        result.push(...group);
    }
    result.push(...singletons);

    return result;
}

/**
 * Compute X offsets for components given their display order.
 * Returns a map from component index to its X offset in pixels.
 */
export function computeComponentOffsets(
    sortedComponents: number[],
    componentSize: number,
    componentGap: number,
): Record<number, number> {
    const offsets: Record<number, number> = {};
    for (let i = 0; i < sortedComponents.length; i++) {
        offsets[sortedComponents[i]] = i * (componentSize + componentGap);
    }
    return offsets;
}

export type ClusterSpan = {
    clusterId: number;
    layer: string;
    seqIdx: number;
    xStart: number;
    xEnd: number;
    y: number;
};

/**
 * Compute horizontal spans for cluster bars.
 * For each (layer, seqIdx), finds contiguous runs of components in the same cluster
 * and returns their bounding box positions.
 */
export function computeClusterSpans(
    sortedComponents: number[],
    layer: string,
    seqIdx: number,
    baseX: number,
    baseY: number,
    componentSize: number,
    offsets: Record<number, number>,
    getClusterId: (layer: string, componentIdx: number) => number | null | undefined,
): ClusterSpan[] {
    const spans: ClusterSpan[] = [];
    if (sortedComponents.length === 0) return spans;

    let currentCluster: number | null = null;
    let spanStartX: number | null = null;
    let spanEndX: number | null = null;

    for (const cIdx of sortedComponents) {
        const clusterId = getClusterId(layer, cIdx);
        const x = baseX + offsets[cIdx];

        if (typeof clusterId === "number") {
            if (clusterId === currentCluster) {
                // Extend current span
                spanEndX = x + componentSize;
            } else {
                // Close previous span if exists
                if (currentCluster !== null && spanStartX !== null && spanEndX !== null) {
                    spans.push({
                        clusterId: currentCluster,
                        layer,
                        seqIdx,
                        xStart: spanStartX,
                        xEnd: spanEndX,
                        y: baseY + componentSize,
                    });
                }
                // Start new span
                currentCluster = clusterId;
                spanStartX = x;
                spanEndX = x + componentSize;
            }
        } else {
            // Close previous span if exists (singleton encountered)
            if (currentCluster !== null && spanStartX !== null && spanEndX !== null) {
                spans.push({
                    clusterId: currentCluster,
                    layer,
                    seqIdx,
                    xStart: spanStartX,
                    xEnd: spanEndX,
                    y: baseY + componentSize,
                });
            }
            currentCluster = null;
            spanStartX = null;
            spanEndX = null;
        }
    }

    // Close final span if exists
    if (currentCluster !== null && spanStartX !== null && spanEndX !== null) {
        spans.push({
            clusterId: currentCluster,
            layer,
            seqIdx,
            xStart: spanStartX,
            xEnd: spanEndX,
            y: baseY + componentSize,
        });
    }

    return spans;
}
