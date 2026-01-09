/**
 * Global display settings using Svelte 5 runes
 */

// Available correlation stat types
export type CorrelationStatType = "pmi" | "bottom_pmi" | "precision" | "recall" | "jaccard";

// Node color mode for graph visualization
export type NodeColorMode = "ci" | "subcomp_act";

export const NODE_COLOR_MODE_LABELS: Record<NodeColorMode, string> = {
    ci: "CI",
    subcomp_act: "Subcomp Act",
};

// Example color mode for activation contexts viewer
export type ExampleColorMode = "ci" | "component_act" | "both";

export const EXAMPLE_COLOR_MODE_LABELS: Record<ExampleColorMode, string> = {
    ci: "CI",
    component_act: "Component Act",
    both: "Both",
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

type StoredSettings = {
    showPmi?: boolean;
    showBottomPmi?: boolean;
    showPrecision?: boolean;
    showRecall?: boolean;
    showJaccard?: boolean;
    showSetOverlapVis?: boolean;
    showEdgeAttributions?: boolean;
    nodeColorMode?: NodeColorMode;
    exampleColorMode?: ExampleColorMode;
};

function loadFromStorage(): StoredSettings | undefined {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        return stored ? (JSON.parse(stored) as StoredSettings) : undefined;
    } catch {
        return undefined;
    }
}

function saveToStorage(settings: Partial<StoredSettings>) {
    try {
        const current = loadFromStorage();
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...current, ...settings }));
    } catch (error) {
        console.error(`Error saving display settings: ${error instanceof Error ? error.message : String(error)}`);
    }
}

const VALID_NODE_COLOR_MODES: NodeColorMode[] = ["ci", "subcomp_act"];
const VALID_EXAMPLE_COLOR_MODES: ExampleColorMode[] = ["ci", "component_act", "both"];

// =============================================================================
// Reactive state (module-level $state)
// =============================================================================

let _showPmi = $state(loadFromStorage()?.showPmi ?? true);
let _showBottomPmi = $state(loadFromStorage()?.showBottomPmi ?? false);
let _showPrecision = $state(loadFromStorage()?.showPrecision ?? true);
let _showRecall = $state(loadFromStorage()?.showRecall ?? true);
let _showJaccard = $state(loadFromStorage()?.showJaccard ?? true);
let _showSetOverlapVis = $state(loadFromStorage()?.showSetOverlapVis ?? true);
let _showEdgeAttributions = $state(loadFromStorage()?.showEdgeAttributions ?? true);
let _nodeColorMode = $state<NodeColorMode>(
    VALID_NODE_COLOR_MODES.includes(loadFromStorage()?.nodeColorMode as NodeColorMode)
        ? (loadFromStorage()?.nodeColorMode as NodeColorMode)
        : "ci",
);
let _exampleColorMode = $state<ExampleColorMode>(
    VALID_EXAMPLE_COLOR_MODES.includes(loadFromStorage()?.exampleColorMode as ExampleColorMode)
        ? (loadFromStorage()?.exampleColorMode as ExampleColorMode)
        : "ci",
);

// =============================================================================
// Exported object with getters/setters
// =============================================================================

export const displaySettings = {
    get showPmi() {
        return _showPmi;
    },
    set showPmi(v: boolean) {
        _showPmi = v;
        saveToStorage({ showPmi: v });
    },

    get showBottomPmi() {
        return _showBottomPmi;
    },
    set showBottomPmi(v: boolean) {
        _showBottomPmi = v;
        saveToStorage({ showBottomPmi: v });
    },

    get showPrecision() {
        return _showPrecision;
    },
    set showPrecision(v: boolean) {
        _showPrecision = v;
        saveToStorage({ showPrecision: v });
    },

    get showRecall() {
        return _showRecall;
    },
    set showRecall(v: boolean) {
        _showRecall = v;
        saveToStorage({ showRecall: v });
    },

    get showJaccard() {
        return _showJaccard;
    },
    set showJaccard(v: boolean) {
        _showJaccard = v;
        saveToStorage({ showJaccard: v });
    },

    get showSetOverlapVis() {
        return _showSetOverlapVis;
    },
    set showSetOverlapVis(v: boolean) {
        _showSetOverlapVis = v;
        saveToStorage({ showSetOverlapVis: v });
    },

    get showEdgeAttributions() {
        return _showEdgeAttributions;
    },
    set showEdgeAttributions(v: boolean) {
        _showEdgeAttributions = v;
        saveToStorage({ showEdgeAttributions: v });
    },

    get nodeColorMode() {
        return _nodeColorMode;
    },
    set nodeColorMode(v: NodeColorMode) {
        _nodeColorMode = v;
        saveToStorage({ nodeColorMode: v });
    },

    get exampleColorMode() {
        return _exampleColorMode;
    },
    set exampleColorMode(v: ExampleColorMode) {
        _exampleColorMode = v;
        saveToStorage({ exampleColorMode: v });
    },

    get hasAnyCorrelationStats() {
        return _showPmi || _showBottomPmi || _showPrecision || _showRecall || _showJaccard;
    },
};
