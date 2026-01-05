/**
 * Global display settings store
 */

import { SvelteSet } from "svelte/reactivity";

// Available correlation stat types
export type CorrelationStatType = "pmi" | "bottom_pmi" | "precision" | "recall" | "jaccard";

// Node color mode for graph visualization
export type NodeColorMode = "ci" | "subcomp_act";

export const NODE_COLOR_MODE_LABELS: Record<NodeColorMode, string> = {
    ci: "CI",
    subcomp_act: "Subcomp Act",
};

export const CORRELATION_STAT_LABELS: Record<CorrelationStatType, string> = {
    pmi: "PMI",
    bottom_pmi: "Bottom PMI",
    precision: "Precision",
    recall: "Recall",
    jaccard: "Jaccard",
};

export const CORRELATION_STAT_DESCRIPTIONS: Record<CorrelationStatType, string> = {
    pmi: "log(P(both) / P(A)P(B))",
    bottom_pmi: "Lowest PMI (anti-correlated)",
    precision: "P(that | this)",
    recall: "P(this | that)",
    jaccard: "Intersection over union",
};

const STORAGE_KEY = "spd-display-settings";
const ALL_STATS: CorrelationStatType[] = ["pmi", "bottom_pmi", "precision", "recall", "jaccard"];
const DEFAULT_ON_STATS: CorrelationStatType[] = ["pmi", "precision", "recall", "jaccard"];

type StoredSettings = {
    visibleCorrelationStats?: string[];
    showSetOverlapVis?: boolean;
    showEdgeAttributions?: boolean;
    nodeColorMode?: NodeColorMode;
};

function loadFromStorage(): StoredSettings | undefined {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? (JSON.parse(stored) as StoredSettings) : undefined;
}

function loadCorrelationStats(): CorrelationStatType[] {
    const stored = loadFromStorage();
    if (stored == null) return DEFAULT_ON_STATS;
    // if any invalid stats are present, delete the key from storage
    if (stored.visibleCorrelationStats?.some((s: string) => !ALL_STATS.includes(s as CorrelationStatType))) {
        stored.visibleCorrelationStats = DEFAULT_ON_STATS;
        saveToStorage(stored);
        return DEFAULT_ON_STATS;
    }
    return stored.visibleCorrelationStats as CorrelationStatType[];
}

function loadShowSetOverlapVis(): boolean {
    return loadFromStorage()?.showSetOverlapVis ?? true;
}

function loadShowEdgeAttributions(): boolean {
    return loadFromStorage()?.showEdgeAttributions ?? true;
}

const VALID_COLOR_MODES: NodeColorMode[] = ["ci", "subcomp_act"];

function loadNodeColorMode(): NodeColorMode {
    const stored = loadFromStorage();
    const mode = stored?.nodeColorMode;
    if (mode == null || !VALID_COLOR_MODES.includes(mode)) {
        return "ci";
    }
    return mode;
}

function saveToStorage(settings: StoredSettings) {
    try {
        const current = loadFromStorage();
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...current, ...settings }));
    } catch (error) {
        console.error(
            `Error saving display settings to storage: ${error instanceof Error ? error.message : String(error)}`,
        );
    }
}

class DisplaySettingsState {
    // Which correlation stats to show (loaded from storage or all enabled by default)
    visibleCorrelationStats = new SvelteSet<CorrelationStatType>(loadCorrelationStats());

    // Whether to show set overlap visualizations
    showSetOverlapVis = $state(loadShowSetOverlapVis());

    // Whether to show edge attribution lists in hover panel
    showEdgeAttributions = $state(loadShowEdgeAttributions());

    // Node color mode for graph visualization
    nodeColorMode = $state<NodeColorMode>(loadNodeColorMode());

    toggleCorrelationStat(stat: CorrelationStatType) {
        if (this.visibleCorrelationStats.has(stat)) {
            // Don't allow disabling the last stat
            if (this.visibleCorrelationStats.size > 1) {
                this.visibleCorrelationStats.delete(stat);
            }
        } else {
            this.visibleCorrelationStats.add(stat);
        }
        saveToStorage({ visibleCorrelationStats: [...this.visibleCorrelationStats] });
    }

    isCorrelationStatVisible(stat: CorrelationStatType): boolean {
        return this.visibleCorrelationStats.has(stat);
    }

    toggleSetOverlapVis() {
        this.showSetOverlapVis = !this.showSetOverlapVis;
        saveToStorage({ showSetOverlapVis: this.showSetOverlapVis });
    }

    toggleEdgeAttributions() {
        this.showEdgeAttributions = !this.showEdgeAttributions;
        saveToStorage({ showEdgeAttributions: this.showEdgeAttributions });
    }

    setNodeColorMode(mode: NodeColorMode) {
        this.nodeColorMode = mode;
        saveToStorage({ nodeColorMode: mode });
    }
}

// Singleton instance
export const displaySettings = new DisplaySettingsState();
