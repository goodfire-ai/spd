/**
 * Run-scoped state hook
 *
 * Call useRun() in App.svelte and provide via context.
 * Child components access it via getContext('run').
 */

import type { Loadable } from ".";
import * as api from "./api";
import type { RunState as RunData, Interpretation } from "./api";
import type { ActivationContextsSummary, ComponentDetail, PromptPreview, TokenInfo } from "./promptAttributionsTypes";

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
    | { status: "generated"; data: Interpretation }
    | { status: "generation-error"; error: unknown };

export function useRun() {
    /** The currently loaded run */
    let run = $state<Loadable<RunData>>({ status: "uninitialized" });

    /** Interpretation labels keyed by component key (layer:cIdx) */
    let interpretations = $state<Loadable<Record<string, InterpretationBackendState>>>({ status: "uninitialized" });

    /** Cluster mapping for the current run */
    let clusterMapping = $state<ClusterMapping | null>(null);

    /** Available prompts for the current run */
    let prompts = $state<Loadable<PromptPreview[]>>({ status: "uninitialized" });

    /** All tokens in the tokenizer for the current run */
    let allTokens = $state<Loadable<TokenInfo[]>>({ status: "uninitialized" });

    /** Activation contexts summary */
    let activationContextsSummary = $state<Loadable<ActivationContextsSummary>>({ status: "uninitialized" });

    /** Cached component details keyed by component key (layer:cIdx) - non-reactive */
    let _componentDetailsCache: Record<string, ComponentDetail> = {};

    /** Reset all run-scoped state */
    function resetRunScopedState() {
        prompts = { status: "uninitialized" };
        allTokens = { status: "uninitialized" };
        interpretations = { status: "uninitialized" };
        activationContextsSummary = { status: "uninitialized" };
        _componentDetailsCache = {};
        clusterMapping = null;
    }

    /** Fetch run-scoped data that can load asynchronously (prompts, interpretations) */
    function fetchRunScopedData() {
        prompts = { status: "loading" };
        interpretations = { status: "loading" };

        api.listPrompts()
            .then((p) => (prompts = { status: "loaded", data: p }))
            .catch((error) => (prompts = { status: "error", error }));
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
            const [status] = await Promise.all([api.getStatus(), fetchTokens()]);
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
                // Fetch tokens if we don't have them (e.g., page refresh)
                if (allTokens.status === "uninitialized") {
                    await fetchTokens();
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

    /** Set interpretation for a component (updates cache without full reload) */
    function setInterpretation(componentKey: string, interpretation: InterpretationBackendState) {
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
        loadRun,
        clearRun,
        syncStatus,
        refreshPrompts,
        getInterpretation,
        setInterpretation,
        getComponentDetail,
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
