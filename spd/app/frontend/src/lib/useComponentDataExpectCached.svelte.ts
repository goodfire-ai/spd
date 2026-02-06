/**
 * Hook for reading component data from prefetched cache.
 *
 * prefetchComponentData() must be called before using this hook.
 * Activation contexts, correlations, and token stats are read from cache.
 * Dataset attributions and interpretation details are fetched on-demand.
 */

import { getContext } from "svelte";
import type { Loadable } from ".";
import {
    ApiError,
    getActivationContextDetail,
    getComponentAttributions,
    getInterpretationDetail,
    requestComponentInterpretation,
} from "./api";
import type { ComponentAttributions, InterpretationDetail } from "./api";
import type {
    SubcomponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "./promptAttributionsTypes";
import { RUN_KEY, type InterpretationBackendState, type RunContext } from "./useRun.svelte";

const DATASET_ATTRIBUTIONS_TOP_K = 20;
/** Fetch more activation examples in background after initial cached load */
const ACTIVATION_EXAMPLES_FULL_LIMIT = 200;

export type { ComponentAttributions as DatasetAttributions };

export type ComponentCoords = { layer: string; cIdx: number };

export function useComponentDataExpectCached() {
    const runState = getContext<RunContext>(RUN_KEY);

    let componentDetail = $state<Loadable<SubcomponentActivationContexts>>({ status: "uninitialized" });
    let correlations = $state<Loadable<SubcomponentCorrelationsResponse | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStatsResponse | null>>({ status: "uninitialized" });
    let datasetAttributions = $state<Loadable<ComponentAttributions | null>>({ status: "uninitialized" });
    let interpretationDetail = $state<Loadable<InterpretationDetail | null>>({ status: "uninitialized" });

    let currentCoords = $state<ComponentCoords | null>(null);
    let requestId = 0;

    function load(layer: string, cIdx: number) {
        currentCoords = { layer, cIdx };
        const thisRequestId = ++requestId;
        const componentKey = `${layer}:${cIdx}`;

        const isStale = () => requestId !== thisRequestId;

        const cachedDetail = runState.expectCachedComponentDetail(componentKey);
        componentDetail = { status: "loaded", data: cachedDetail };
        correlations = { status: "loaded", data: runState.expectCachedCorrelations(componentKey) };
        tokenStats = { status: "loaded", data: runState.expectCachedTokenStats(componentKey) };

        // Fetch more activation examples in background (overwrites cached data when complete)
        getActivationContextDetail(layer, cIdx, ACTIVATION_EXAMPLES_FULL_LIMIT)
            .then((data) => {
                if (isStale()) return;
                // Only update if we got more examples than cached
                if (data.example_tokens.length > cachedDetail.example_tokens.length) {
                    componentDetail = { status: "loaded", data };
                }
            })
            .catch((error) => {
                if (isStale()) return;
                componentDetail = { status: "error", error };
            });

        // Skip fetch entirely if dataset attributions not available for this run
        if (runState.datasetAttributionsAvailable) {
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
        } else {
            datasetAttributions = { status: "loaded", data: null };
        }

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
