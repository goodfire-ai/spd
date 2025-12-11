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

function loadFromStorage(): CorrelationStatType[] {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            const parsed = JSON.parse(stored);
            // Validate that all items are valid stat types
            const valid = parsed.visibleCorrelationStats?.filter((s: string) =>
                ALL_STATS.includes(s as CorrelationStatType)
            );
            if (valid?.length > 0) return valid;
        }
    } catch {
        // Ignore parse errors
    }
    return ALL_STATS;
}

function saveToStorage(stats: SvelteSet<CorrelationStatType>) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ visibleCorrelationStats: [...stats] }));
    } catch {
        // Ignore storage errors
    }
}

// View settings state
class ViewSettings {
    // Which correlation stats to show (loaded from storage or all enabled by default)
    visibleCorrelationStats = new SvelteSet<CorrelationStatType>(loadFromStorage());

    toggleCorrelationStat(stat: CorrelationStatType) {
        if (this.visibleCorrelationStats.has(stat)) {
            // Don't allow disabling the last stat
            if (this.visibleCorrelationStats.size > 1) {
                this.visibleCorrelationStats.delete(stat);
            }
        } else {
            this.visibleCorrelationStats.add(stat);
        }
        saveToStorage(this.visibleCorrelationStats);
    }

    isCorrelationStatVisible(stat: CorrelationStatType): boolean {
        return this.visibleCorrelationStats.has(stat);
    }
}

// Singleton instance
export const viewSettings = new ViewSettings();
