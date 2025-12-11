/**
 * Global view settings store for controlling which statistics are visible
 */

import { SvelteSet } from "svelte/reactivity";

// Available correlation stat types
export type CorrelationStatType = "pmi" | "precision" | "recall" | "f1" | "jaccard";

export const CORRELATION_STAT_LABELS: Record<CorrelationStatType, string> = {
    pmi: "PMI",
    precision: "Precision",
    recall: "Recall",
    f1: "F1",
    jaccard: "Jaccard",
};

export const CORRELATION_STAT_DESCRIPTIONS: Record<CorrelationStatType, string> = {
    pmi: "log(P(both) / P(A)P(B))",
    precision: "P(that | this)",
    recall: "P(this | that)",
    f1: "Harmonic mean of precision and recall",
    jaccard: "Intersection over union",
};

const STORAGE_KEY = "spd-view-settings";
const ALL_STATS: CorrelationStatType[] = ["pmi", "precision", "recall", "f1", "jaccard"];

type StoredSettings = {
    visibleCorrelationStats?: string[];
    showSetOverlapVis?: boolean;
};

function loadFromStorage(): StoredSettings {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? (JSON.parse(stored) as StoredSettings) : {};
}

function loadCorrelationStats(): CorrelationStatType[] {
    const stored = loadFromStorage();
    const valid = stored.visibleCorrelationStats?.filter((s: string) => ALL_STATS.includes(s as CorrelationStatType));
    if (valid && valid.length > 0) return valid as CorrelationStatType[];
    return ALL_STATS;
}

function loadShowSetOverlapVis(): boolean {
    const stored = loadFromStorage();
    return stored.showSetOverlapVis ?? true;
}

function saveToStorage(settings: StoredSettings) {
    try {
        const current = loadFromStorage();
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...current, ...settings }));
    } catch {
        // Ignore storage errors
    }
}

// View settings state
class ViewSettings {
    // Which correlation stats to show (loaded from storage or all enabled by default)
    visibleCorrelationStats = new SvelteSet<CorrelationStatType>(loadCorrelationStats());

    // Whether to show set overlap visualizations
    showSetOverlapVis = $state(loadShowSetOverlapVis());

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
}

// Singleton instance
export const viewSettings = new ViewSettings();
