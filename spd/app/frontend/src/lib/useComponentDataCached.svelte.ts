/**
 * Cached version of useComponentData that reads from prefetched cache.
 *
 * IMPORTANT: prefetchComponentData() must be called before using this hook,
 * otherwise it will throw on cache miss.
 *
 * This hook provides instant access to component data without network requests.
 * Data for activation contexts, correlations, and token stats is read from cache.
 * Dataset attributions and interpretation details are still fetched on-demand.
 */

import { getContext } from "svelte";
import type { Loadable } from ".";
import { ApiError, getComponentAttributions, getInterpretationDetail, requestComponentInterpretation } from "./api";
import type { ComponentAttributions, InterpretationDetail } from "./api";
import type {
    ComponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "./promptAttributionsTypes";
import { RUN_KEY, type InterpretationBackendState, type RunContext } from "./useRun.svelte";

/** Dataset attributions top-k */
const DATASET_ATTRIBUTIONS_TOP_K = 20;

export type { ComponentAttributions as DatasetAttributions };

export type ComponentCoords = { layer: string; cIdx: number };

/**
 * Cached hook for loading component data from prefetched cache.
 *
 * Reads activation contexts, correlations, and token stats synchronously from cache.
 * Dataset attributions and interpretation details are fetched on-demand.
 */
export function useComponentDataCached() {
    const runState = getContext<RunContext>(RUN_KEY);

    // These are read synchronously from cache
    let componentDetail = $state<Loadable<SubcomponentActivationContexts>>({ status: "uninitialized" });
    let correlations = $state<Loadable<ComponentCorrelationsResponse | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStatsResponse | null>>({ status: "uninitialized" });

    // These are still fetched on-demand
    let datasetAttributions = $state<Loadable<ComponentAttributions | null>>({ status: "uninitialized" });
    let interpretationDetail = $state<Loadable<InterpretationDetail | null>>({ status: "uninitialized" });

    let currentCoords = $state<ComponentCoords | null>(null);
    let requestId = 0;

    function load(layer: string, cIdx: number) {
        currentCoords = { layer, cIdx };
        const thisRequestId = ++requestId;
        const componentKey = `${layer}:${cIdx}`;

        const isStale = () => requestId !== thisRequestId;

        // Read cached data synchronously (throws if not prefetched)
        try {
            componentDetail = { status: "loaded", data: runState.getCachedComponentDetail(componentKey) };
        } catch (e) {
            componentDetail = { status: "error", error: e };
        }

        try {
            correlations = { status: "loaded", data: runState.getCachedCorrelations(componentKey) };
        } catch {
            // Correlations may not exist for all components
            correlations = { status: "loaded", data: null };
        }

        try {
            tokenStats = { status: "loaded", data: runState.getCachedTokenStats(componentKey) };
        } catch {
            // Token stats may not exist for all components
            tokenStats = { status: "loaded", data: null };
        }

        // Fetch dataset attributions on-demand (not cached)
        datasetAttributions = { status: "loading" };
        getComponentAttributions(layer, cIdx, DATASET_ATTRIBUTIONS_TOP_K)
            .then((data) => {
                if (isStale()) return;
                datasetAttributions = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    datasetAttributions = { status: "loaded", data: null };
                } else {
                    datasetAttributions = { status: "error", error };
                }
            });

        // Fetch interpretation detail on-demand (not cached)
        interpretationDetail = { status: "loading" };
        getInterpretationDetail(layer, cIdx)
            .then((data) => {
                if (isStale()) return;
                interpretationDetail = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    interpretationDetail = { status: "loaded", data: null };
                } else {
                    interpretationDetail = { status: "error", error };
                }
            });
    }

    function reset() {
        requestId++;
        currentCoords = null;
        componentDetail = { status: "uninitialized" };
        correlations = { status: "uninitialized" };
        tokenStats = { status: "uninitialized" };
        datasetAttributions = { status: "uninitialized" };
        interpretationDetail = { status: "uninitialized" };
    }

    // Interpretation is derived from the global cache
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

            // Fetch the detail now that it exists
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
