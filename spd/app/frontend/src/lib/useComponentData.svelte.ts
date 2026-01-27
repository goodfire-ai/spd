import { getContext } from "svelte";
import type { Loadable } from ".";
import {
    ApiError,
    getComponentAttributions,
    getComponentCorrelations,
    getComponentTokenStats,
    getInterpretationDetail,
    requestComponentInterpretation,
} from "./api";
import type { ComponentAttributions, InterpretationDetail } from "./api";
import type {
    ComponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "./promptAttributionsTypes";
import { RUN_KEY, type InterpretationBackendState, type RunContext } from "./useRun.svelte";

/** Correlations are paginated in the UI, so fetch more */
const CORRELATIONS_TOP_K = 100;
/** Token stats are displayed directly (max 50 shown) */
const TOKEN_STATS_TOP_K = 50;
/** Dataset attributions top-k */
const DATASET_ATTRIBUTIONS_TOP_K = 20;

/** Enable performance timing logs */
const PERF_TIMING_ENABLED = true;

type TimingData = {
    loadStartTime: number;
    componentDetail: number | null;
    correlations: number | null;
    tokenStats: number | null;
    datasetAttributions: number | null;
    interpretationDetail: number | null;
    allComplete: number | null;
};

/** Performance mark helper - creates marks visible in DevTools Performance panel */
function perfMark(name: string, componentKey: string) {
    if (PERF_TIMING_ENABLED) {
        performance.mark(`${name}:${componentKey}`);
    }
}

/** Measure time between two marks */
function perfMeasure(name: string, startMark: string, endMark: string) {
    if (PERF_TIMING_ENABLED) {
        try {
            performance.measure(name, startMark, endMark);
        } catch {
            // Marks may not exist if component unmounted early
        }
    }
}

export type { ComponentAttributions as DatasetAttributions };

export type ComponentCoords = { layer: string; cIdx: number };

/**
 * Hook for loading component data (detail, correlations, token stats, interpretation detail).
 *
 * Call `load(layer, cIdx)` explicitly when you want to fetch data.
 * Interpretation headline is derived from the global runState cache.
 * Interpretation detail (reasoning + prompt) is fetched on-demand.
 */
export function useComponentData() {
    const runState = getContext<RunContext>(RUN_KEY);

    let componentDetail = $state<Loadable<SubcomponentActivationContexts>>({ status: "uninitialized" });
    // null inside Loadable means "no data for this component" (404)
    let correlations = $state<Loadable<ComponentCorrelationsResponse | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStatsResponse | null>>({ status: "uninitialized" });
    let datasetAttributions = $state<Loadable<ComponentAttributions | null>>({ status: "uninitialized" });

    let interpretationDetail = $state<Loadable<InterpretationDetail | null>>({ status: "uninitialized" });

    // Current coords being loaded/displayed (for interpretation lookup)
    let currentCoords = $state<ComponentCoords | null>(null);

    // Request counter for handling stale responses
    let requestId = 0;

    /**
     * Load all data for the given component.
     * Call this from event handlers or on mount.
     */
    function load(layer: string, cIdx: number) {
        currentCoords = { layer, cIdx };
        const thisRequestId = ++requestId;
        const componentKey = `${layer}:${cIdx}`;

        // Performance marks for lifecycle tracking
        perfMark("fetch-start", componentKey);

        // Set loading states
        componentDetail = { status: "loading" };
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };
        datasetAttributions = { status: "loading" };
        interpretationDetail = { status: "loading" };

        // Helper to check if this request is still current
        const isStale = () => requestId !== thisRequestId;

        // Performance timing
        const timing: TimingData = {
            loadStartTime: performance.now(),
            componentDetail: null,
            correlations: null,
            tokenStats: null,
            datasetAttributions: null,
            interpretationDetail: null,
            allComplete: null,
        };
        let pendingCount = 5;

        function checkAllComplete() {
            pendingCount--;
            if (pendingCount === 0 && !isStale()) {
                perfMark("fetch-end", componentKey);
                perfMeasure(`fetch-total:${componentKey}`, `fetch-start:${componentKey}`, `fetch-end:${componentKey}`);

                if (PERF_TIMING_ENABLED) {
                    timing.allComplete = performance.now() - timing.loadStartTime;
                    console.log(
                        `[ComponentData ${componentKey}] All fetches complete in ${timing.allComplete.toFixed(0)}ms`,
                        `\n  componentDetail: ${timing.componentDetail?.toFixed(0) ?? "pending"}ms`,
                        `\n  correlations: ${timing.correlations?.toFixed(0) ?? "pending"}ms`,
                        `\n  tokenStats: ${timing.tokenStats?.toFixed(0) ?? "pending"}ms`,
                        `\n  datasetAttributions: ${timing.datasetAttributions?.toFixed(0) ?? "pending"}ms`,
                        `\n  interpretationDetail: ${timing.interpretationDetail?.toFixed(0) ?? "pending"}ms`,
                    );
                }
            }
        }

        // Fetch component detail (cached in runState after first call)
        runState
            .getActivationContextDetail(layer, cIdx)
            .then((data) => {
                timing.componentDetail = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                componentDetail = { status: "loaded", data };
            })
            .catch((error) => {
                timing.componentDetail = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                componentDetail = { status: "error", error };
            })
            .finally(checkAllComplete);

        // Fetch correlations (404 = no data for this component)
        getComponentCorrelations(layer, cIdx, CORRELATIONS_TOP_K)
            .then((data) => {
                timing.correlations = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                correlations = { status: "loaded", data };
            })
            .catch((error) => {
                timing.correlations = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    correlations = { status: "loaded", data: null };
                } else {
                    correlations = { status: "error", error };
                }
            })
            .finally(checkAllComplete);

        // Fetch token stats (404 = no data for this component)
        getComponentTokenStats(layer, cIdx, TOKEN_STATS_TOP_K)
            .then((data) => {
                timing.tokenStats = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                tokenStats = { status: "loaded", data };
            })
            .catch((error) => {
                timing.tokenStats = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    tokenStats = { status: "loaded", data: null };
                } else {
                    tokenStats = { status: "error", error };
                }
            })
            .finally(checkAllComplete);

        // Fetch dataset attributions (404 = not available)
        getComponentAttributions(layer, cIdx, DATASET_ATTRIBUTIONS_TOP_K)
            .then((data) => {
                timing.datasetAttributions = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                datasetAttributions = { status: "loaded", data };
            })
            .catch((error) => {
                timing.datasetAttributions = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    datasetAttributions = { status: "loaded", data: null };
                } else {
                    datasetAttributions = { status: "error", error };
                }
            })
            .finally(checkAllComplete);

        // Fetch interpretation detail (404 = no interpretation for this component)
        getInterpretationDetail(layer, cIdx)
            .then((data) => {
                timing.interpretationDetail = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                interpretationDetail = { status: "loaded", data };
            })
            .catch((error) => {
                timing.interpretationDetail = performance.now() - timing.loadStartTime;
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    interpretationDetail = { status: "loaded", data: null };
                } else {
                    interpretationDetail = { status: "error", error };
                }
            })
            .finally(checkAllComplete);
    }

    /**
     * Reset all state to uninitialized.
     */
    function reset() {
        requestId++; // Invalidate any in-flight requests
        currentCoords = null;
        componentDetail = { status: "uninitialized" };
        correlations = { status: "uninitialized" };
        tokenStats = { status: "uninitialized" };
        datasetAttributions = { status: "uninitialized" };
        interpretationDetail = { status: "uninitialized" };
    }

    // Interpretation is derived from the global cache - reactive to both coords and cache
    const interpretation = $derived.by((): Loadable<InterpretationBackendState> => {
        if (!currentCoords) return { status: "uninitialized" };
        return runState.getInterpretation(`${currentCoords.layer}:${currentCoords.cIdx}`);
    });

    async function generateInterpretation() {
        if (!currentCoords) return;

        const { layer, cIdx } = currentCoords;
        const componentKey = `${layer}:${cIdx}`;

        try {
            runState.setInterpretation(componentKey, { status: "generating" });
            const result = await requestComponentInterpretation(layer, cIdx);
            runState.setInterpretation(componentKey, { status: "generated", data: result });

            // Fetch the detail (reasoning + prompt) now that it exists
            try {
                const detail = await getInterpretationDetail(layer, cIdx);
                interpretationDetail = { status: "loaded", data: detail };
            } catch (detailError) {
                interpretationDetail = { status: "error", error: detailError };
            }
        } catch (e) {
            runState.setInterpretation(componentKey, {
                status: "generation-error",
                error: e instanceof Error ? e.message : String(e),
            });
        }
    }

    return {
        get componentDetail() {
            return componentDetail;
        },
        get correlations() {
            return correlations;
        },
        get tokenStats() {
            return tokenStats;
        },
        get datasetAttributions() {
            return datasetAttributions;
        },
        get interpretation() {
            return interpretation;
        },
        get interpretationDetail() {
            return interpretationDetail;
        },
        load,
        reset,
        generateInterpretation,
    };
}
