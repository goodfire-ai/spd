/**
 * Run-scoped state hook
 *
 * Call useRun() in App.svelte and provide via context.
 * Child components access it via getContext('run').
 */

import type { Loadable } from ".";
import * as api from "./api";
import type { LoadedRun as RunData, InterpretationHeadline, ModelInfo } from "./api";
import type {
    SubcomponentCorrelationsResponse,
    PromptPreview,
    SubcomponentActivationContexts,
    TokenInfo,
    TokenStatsResponse,
    SubcomponentMetadata,
} from "./promptAttributionsTypes";

/** Maps component keys to cluster IDs. Singletons (unclustered components) have null values. */
export type ClusterMappingData = Record<string, number | null>;

type ClusterMapping = {
    data: ClusterMappingData;
    filePath: string;
    runWandbPath: string;
};

/**
 * Per-component interpretation status. Nested inside Loadable<> because:
 * - Outer Loadable tracks fetching the interpretations cache from the server
 * - Inner state tracks each component's interpretation (none/generating/generated/error)
 * These are distinct concerns; conflating them would lose semantic clarity.
 */
export type InterpretationBackendState =
    | { status: "none" }
    | { status: "generating" }
    | { status: "generated"; data: InterpretationHeadline }
    | { status: "generation-error"; error: unknown };

export function useRun() {
    /** The currently loaded run */
    let run = $state<Loadable<RunData>>({ status: "uninitialized" });

    /** Interpretation labels keyed by component key (layer:cIdx) */
    let interpretations = $state<Loadable<Record<string, InterpretationBackendState>>>({ status: "uninitialized" });

    /** Intruder eval scores keyed by component key */
    let intruderScores = $state<Loadable<Record<string, number>>>({ status: "uninitialized" });

    /** Cluster mapping for the current run */
    let clusterMapping = $state<ClusterMapping | null>(null);

    /** Available prompts for the current run */
    let prompts = $state<Loadable<PromptPreview[]>>({ status: "uninitialized" });

    /** All tokens in the tokenizer for the current run */
    let allTokens = $state<Loadable<TokenInfo[]>>({ status: "uninitialized" });

    /** Model topology info for frontend layout */
    let modelInfo = $state<ModelInfo | null>(null);

    /** Activation contexts summary */
    let activationContextsSummary = $state<Loadable<Record<string, SubcomponentMetadata[]>>>({
        status: "uninitialized",
    });

    // Cached component data keyed by component key (layer:cIdx) - non-reactive
    let _componentDetailsCache: Record<string, SubcomponentActivationContexts> = {};
    let _correlationsCache: Record<string, SubcomponentCorrelationsResponse> = {};
    let _tokenStatsCache: Record<string, TokenStatsResponse> = {};

    // Prefetch parameters for bulk component data
    const PREFETCH_ACTIVATION_CONTEXTS_LIMIT = 100;
    const PREFETCH_CORRELATIONS_TOP_K = 20;
    const PREFETCH_TOKEN_STATS_TOP_K = 30;

    /** Reset all run-scoped state */
    function resetRunScopedState() {
        prompts = { status: "uninitialized" };
        allTokens = { status: "uninitialized" };
        interpretations = { status: "uninitialized" };
        intruderScores = { status: "uninitialized" };
        activationContextsSummary = { status: "uninitialized" };
        _componentDetailsCache = {};
        _correlationsCache = {};
        _tokenStatsCache = {};
        clusterMapping = null;
        modelInfo = null;
    }

    /** Fetch run-scoped data that can load asynchronously (prompts, interpretations) */
    function fetchRunScopedData() {
        prompts = { status: "loading" };
        interpretations = { status: "loading" };
        intruderScores = { status: "loading" };

        api.listPrompts()
            .then((p) => (prompts = { status: "loaded", data: p }))
            .catch((error) => (prompts = { status: "error", error }));
        api.getIntruderScores()
            .then((data) => (intruderScores = { status: "loaded", data }))
            .catch((error) => (intruderScores = { status: "error", error }));
        api.getAllInterpretations()
            .then((i) => {
                interpretations = {
                    status: "loaded",
                    data: Object.fromEntries(
                        Object.entries(i).map(([key, interpretation]): [string, InterpretationBackendState] => [
                            key,
                            {
                                status: "generated",
                                data: interpretation,
                            },
                        ]),
                    ),
                };
            })
            .catch((error) => (interpretations = { status: "error", error }));
    }

    /** Fetch tokens - must complete before run is considered loaded */
    async function fetchTokens(): Promise<TokenInfo[]> {
        allTokens = { status: "loading" };
        const tokens = await api.getAllTokens();
        allTokens = { status: "loaded", data: tokens };
        return tokens;
    }

    async function loadRun(wandbPath: string, contextLength: number) {
        run = { status: "loading" };
        try {
            await api.loadRun(wandbPath, contextLength);
            const [status, info] = await Promise.all([api.getStatus(), fetchTokens(), api.getModelInfo()]).then(
                ([s, , m]) => [s, m] as const,
            );
            modelInfo = info;
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
            if (status) {
                // Fetch tokens and model info if we don't have them (e.g., page refresh)
                if (allTokens.status === "uninitialized") {
                    await fetchTokens();
                }
                if (modelInfo === null) {
                    modelInfo = await api.getModelInfo();
                }
                run = { status: "loaded", data: status };
                // Fetch other run-scoped data if we don't have it
                if (interpretations.status === "uninitialized") {
                    fetchRunScopedData();
                }
            } else if (run.status === "loaded") {
                run = { status: "error", error: "Backend state lost" };
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
    function getInterpretation(componentKey: string): Loadable<InterpretationBackendState> {
        switch (interpretations.status) {
            case "uninitialized":
                return { status: "uninitialized" };
            case "loading":
                return { status: "loading" };
            case "error":
                return { status: "error", error: interpretations.error };
            case "loaded":
                return { status: "loaded", data: interpretations.data[componentKey] ?? { status: "none" } };
        }
    }

    /** Get intruder score for a component, if available */
    function getIntruderScore(componentKey: string): number | null {
        if (intruderScores.status !== "loaded") return null;
        return intruderScores.data[componentKey] ?? null;
    }

    /** Set interpretation for a component (updates cache without full reload) */
    function setInterpretation(componentKey: string, interpretation: InterpretationBackendState) {
        if (interpretations.status === "loaded") {
            interpretations.data[componentKey] = interpretation;
        }
    }

    /** Get activation context detail (fetches once, then cached) */
    async function getActivationContextDetail(layer: string, cIdx: number): Promise<SubcomponentActivationContexts> {
        const cacheKey = `${layer}:${cIdx}`;
        if (cacheKey in _componentDetailsCache) return _componentDetailsCache[cacheKey];

        const detail = await api.getActivationContextDetail(layer, cIdx);
        _componentDetailsCache[cacheKey] = detail;
        return detail;
    }

    /**
     * Bulk prefetch component data for all given component keys.
     * Uses a single combined endpoint to avoid GIL contention from concurrent requests.
     */
    async function prefetchComponentData(componentKeys: string[]): Promise<void> {
        if (componentKeys.length === 0) return;

        const response = await api.getComponentDataBulk(
            componentKeys,
            PREFETCH_ACTIVATION_CONTEXTS_LIMIT,
            PREFETCH_CORRELATIONS_TOP_K,
            PREFETCH_TOKEN_STATS_TOP_K,
        );

        Object.assign(_componentDetailsCache, response.activation_contexts);
        Object.assign(_correlationsCache, response.correlations);
        Object.assign(_tokenStatsCache, response.token_stats);
    }

    /**
     * Read cached component detail. Throws if not prefetched.
     */
    function expectCachedComponentDetail(componentKey: string): SubcomponentActivationContexts {
        const cached = _componentDetailsCache[componentKey];
        if (!cached) throw new Error(`Component detail not prefetched: ${componentKey}`);
        return cached;
    }

    /**
     * Read cached correlations.
     * Returns null if component has no correlation data (e.g., rarely-firing components).
     */
    function expectCachedCorrelations(componentKey: string): SubcomponentCorrelationsResponse | null {
        return _correlationsCache[componentKey] ?? null;
    }

    /**
     * Read cached token stats.
     * Returns null if component has no token stats (e.g., rarely-firing components).
     */
    function expectCachedTokenStats(componentKey: string): TokenStatsResponse | null {
        return _tokenStatsCache[componentKey] ?? null;
    }

    /** Load activation contexts summary (fire-and-forget, updates state) */
    function loadActivationContextsSummary() {
        if (activationContextsSummary.status === "loaded" || activationContextsSummary.status === "loading") return;

        activationContextsSummary = { status: "loading" };
        api.getActivationContextsSummary()
            .then((data) => (activationContextsSummary = { status: "loaded", data }))
            .catch((error) => (activationContextsSummary = { status: "error", error }));
    }

    /** Set cluster mapping for the current run */
    function setClusterMapping(data: ClusterMappingData, filePath: string, runWandbPath: string) {
        clusterMapping = { data, filePath, runWandbPath };
    }

    /** Clear cluster mapping */
    function clearClusterMapping() {
        clusterMapping = null;
    }

    function getClusterId(layer: string, cIdx: number): number | null {
        const key = `${layer}:${cIdx}`;
        return clusterMapping?.data[key] ?? null;
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
        get activationContextsSummary() {
            return activationContextsSummary;
        },
        get datasetAttributionsAvailable() {
            return run.status === "loaded" && run.data.dataset_attributions_available;
        },
        get modelInfo() {
            return modelInfo;
        },
        loadRun,
        clearRun,
        syncStatus,
        refreshPrompts,
        getInterpretation,
        setInterpretation,
        getIntruderScore,
        getActivationContextDetail,
        prefetchComponentData,
        expectCachedComponentDetail,
        expectCachedCorrelations,
        expectCachedTokenStats,
        loadActivationContextsSummary,
        setClusterMapping,
        clearClusterMapping,
        getClusterId,
    };
}

/** Type of the run context returned by useRun() */
export type RunContext = ReturnType<typeof useRun>;

/** Context key for run state */
export const RUN_KEY = "run";
