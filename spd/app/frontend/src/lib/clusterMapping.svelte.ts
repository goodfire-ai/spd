/**
 * Global cluster mapping store
 *
 * Stores a mapping from component keys (layer:component_idx) to cluster IDs.
 * The mapping is tied to a specific run (wandb_path).
 */

/** Maps component keys to cluster IDs. Singletons (unclustered components) have null values. */
export type ClusterMappingData = Record<string, number | null>;

class ClusterMappingState {
    mapping = $state<ClusterMappingData | null>(null);
    filePath = $state<string | null>(null);
    /** The wandb_path of the run this mapping is associated with */
    runWandbPath = $state<string | null>(null);

    setMapping(mapping: ClusterMappingData, filePath: string, runWandbPath: string) {
        this.mapping = mapping;
        this.filePath = filePath;
        this.runWandbPath = runWandbPath;
    }

    clear() {
        this.mapping = null;
        this.filePath = null;
        this.runWandbPath = null;
    }

    /**
     * Clear the mapping if the run has changed.
     * Call this when the loaded run changes.
     */
    clearIfRunChanged(newRunWandbPath: string | null) {
        if (this.runWandbPath !== null && this.runWandbPath !== newRunWandbPath) {
            this.clear();
        }
    }

    /**
     * Get the cluster ID for a component.
     * Returns:
     * - undefined: no mapping loaded or key not in mapping
     * - null: singleton (component not in any cluster)
     * - number: cluster ID
     */
    getClusterId(layer: string, componentIdx: number): number | null | undefined {
        if (!this.mapping) return undefined;
        const key = `${layer}:${componentIdx}`;
        if (!(key in this.mapping)) return undefined;
        return this.mapping[key];
    }
}

export const clusterMapping = new ClusterMappingState();
