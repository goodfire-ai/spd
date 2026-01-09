/**
 * Run-scoped state hook
 *
 * Call useRunState() in App.svelte and provide via context.
 * Child components access it via getContext('runState').
 */

import type { Loadable } from ".";
import * as api from "./api";
import type { RunState as RunData, Interpretation } from "./api";
import type {
    ActivationContextsSummary,
    ComponentDetail,
    PromptPreview,
    TokenInfo,
} from "./localAttributionsTypes";

/** Maps component keys to cluster IDs. Singletons (unclustered components) have null values. */
export type ClusterMappingData = Record<string, number | null>;

type ClusterMapping = {
    data: ClusterMappingData;
    filePath: string;
    runWandbPath: string;
};

export function useRunState() {
    /** The currently loaded run */
    let run = $state<Loadable<RunData>>({ status: "uninitialized" });

    /** Interpretation labels keyed by component key (layer:cIdx) */
    let interpretations = $state<Loadable<Record<string, Interpretation>>>({ status: "uninitialized" });

    /** Cluster mapping for the current run */
    let clusterMapping = $state<ClusterMapping | null>(null);

    /** Available prompts for the current run */
    let prompts = $state<Loadable<PromptPreview[]>>({ status: "uninitialized" });

    /** All tokens in the tokenizer for the current run */
    let allTokens = $state<Loadable<TokenInfo[]>>({ status: "uninitialized" });

    /** Cached component details keyed by component key (layer:cIdx) - non-reactive */
    let _componentDetailsCache: Record<string, ComponentDetail> = {};

    /** Cached activation contexts summary - non-reactive */
    let _summaryCache: ActivationContextsSummary | null = null;

    /** Reset all run-scoped state */
    function resetRunScopedState() {
        prompts = { status: "uninitialized" };
        allTokens = { status: "uninitialized" };
        interpretations = { status: "uninitialized" };
        _componentDetailsCache = {};
        _summaryCache = null;
        clusterMapping = null;
    }

    /** Fetch run-scoped data (prompts, tokens, interpretations) */
    function fetchRunScopedData() {
        prompts = { status: "loading" };
        allTokens = { status: "loading" };
        interpretations = { status: "loading" };

        api.listPrompts()
            .then((p) => (prompts = { status: "loaded", data: p }))
            .catch((error) => (prompts = { status: "error", error }));
        api.getAllTokens()
            .then((t) => (allTokens = { status: "loaded", data: t }))
            .catch((error) => (allTokens = { status: "error", error }));
        api.getAllInterpretations()
            .then((i) => (interpretations = { status: "loaded", data: i }))
            .catch((error) => (interpretations = { status: "error", error }));
    }

    async function loadRun(wandbPath: string, contextLength: number) {
        run = { status: "loading" };
        try {
            await api.loadRun(wandbPath, contextLength);
            const status = await api.getStatus();
            if (status) {
                run = { status: "loaded", data: status };
                fetchRunScopedData();
            } else {
                run = { status: "error", error: "Failed to load run" };
            }
        } catch (error) {
            run = { status: "error", error };
        }
    }

    function clearRun() {
        run = { status: "uninitialized" };
        resetRunScopedState();
    }

    /** Check backend status and sync run state */
    async function syncStatus() {
        try {
            const status = await api.getStatus();
            if (run.status === "loaded" && run.data && !status) {
                run = { status: "error", error: "Backend state lost (restarted)" };
                return;
            }
            if (status) {
                const wasLoaded = run.status === "loaded";
                run = { status: "loaded", data: status };
                // Fetch run-scoped data if we weren't already loaded (e.g., page refresh with backend still running)
                if (!wasLoaded) {
                    fetchRunScopedData();
                }
            } else {
                run = { status: "uninitialized" };
            }
        } catch {
            if (run.status === "loaded") {
                run = { status: "error", error: "Backend unreachable" };
            }
        }
    }

    /** Refresh prompts list (e.g., after generating new prompts) */
    async function refreshPrompts() {
        prompts = { status: "loaded", data: await api.listPrompts() };
    }

    /** Get interpretation for a component, if available */
    function getInterpretation(componentKey: string): Interpretation | undefined {
        if (interpretations.status !== "loaded") return undefined;
        return interpretations.data[componentKey];
    }

    /** Set interpretation for a component (updates cache without full reload) */
    function setInterpretation(componentKey: string, interpretation: Interpretation) {
        if (interpretations.status === "loaded") {
            interpretations.data[componentKey] = interpretation;
        }
    }

    /** Get component detail (fetches once, then cached) */
    async function getComponentDetail(layer: string, cIdx: number): Promise<ComponentDetail> {
        const cacheKey = `${layer}:${cIdx}`;
        if (cacheKey in _componentDetailsCache) return _componentDetailsCache[cacheKey];

        const detail = await api.getComponentDetail(layer, cIdx);
        _componentDetailsCache[cacheKey] = detail;
        return detail;
    }

    /** Get activation contexts summary (fetches once, then cached) */
    async function getActivationContextsSummary(): Promise<ActivationContextsSummary> {
        if (_summaryCache) return _summaryCache;

        const summary = await api.getActivationContextsSummary();
        _summaryCache = summary;
        return summary;
    }

    /** Set cluster mapping for the current run */
    function setClusterMapping(data: ClusterMappingData, filePath: string, runWandbPath: string) {
        clusterMapping = { data, filePath, runWandbPath };
    }

    /** Clear cluster mapping */
    function clearClusterMapping() {
        clusterMapping = null;
    }

    return {
        get run() {
            return run;
        },
        get interpretations() {
            return interpretations;
        },
        get clusterMapping() {
            return clusterMapping;
        },
        get prompts() {
            return prompts;
        },
        get allTokens() {
            return allTokens;
        },
        loadRun,
        clearRun,
        syncStatus,
        refreshPrompts,
        getInterpretation,
        setInterpretation,
        getComponentDetail,
        getActivationContextsSummary,
        setClusterMapping,
        clearClusterMapping,
    };
}

/** Type of the runState returned by useRunState() */
export type RunStateContext = ReturnType<typeof useRunState>;

/** Context key for runState */
export const RUN_STATE_KEY = "runState";
