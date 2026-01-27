/**
 * Bundled version of useComponentData that fetches all data in a single request.
 *
 * This eliminates 4 HTTP roundtrips vs the 5 individual endpoints,
 * saving ~400ms+ over SSH tunnels.
 *
 * Usage: Drop-in replacement for useComponentData when network latency is high.
 */

import { getContext } from "svelte";
import type { Loadable } from ".";
import { getComponentDataBundle, requestComponentInterpretation, getInterpretationDetail } from "./api";
import type { ComponentAttributions, InterpretationDetail } from "./api";
import type { ComponentCorrelationsResponse, SubcomponentActivationContexts, TokenStatsResponse } from "./promptAttributionsTypes";
import { RUN_KEY, type InterpretationBackendState, type RunContext } from "./useRun.svelte";

/** Correlations are paginated in the UI, so fetch more */
const CORRELATIONS_TOP_K = 100;
/** Token stats are displayed directly (max 50 shown) */
const TOKEN_STATS_TOP_K = 50;
/** Dataset attributions top-k */
const DATASET_ATTRIBUTIONS_TOP_K = 20;
/** Default limit for activation examples */
const DETAIL_LIMIT = 30;

/** Enable performance timing logs */
const PERF_TIMING_ENABLED = true;

export type { ComponentAttributions as DatasetAttributions };

export type ComponentCoords = { layer: string; cIdx: number };

/**
 * Bundled hook for loading component data in a single request.
 *
 * Same interface as useComponentData, but uses the bundled endpoint
 * to reduce HTTP roundtrip overhead.
 */
export function useComponentDataBundled() {
    const runState = getContext<RunContext>(RUN_KEY);

    let componentDetail = $state<Loadable<SubcomponentActivationContexts>>({ status: "uninitialized" });
    let correlations = $state<Loadable<ComponentCorrelationsResponse | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStatsResponse | null>>({ status: "uninitialized" });
    let datasetAttributions = $state<Loadable<ComponentAttributions | null>>({ status: "uninitialized" });
    let interpretationDetail = $state<Loadable<InterpretationDetail | null>>({ status: "uninitialized" });

    let currentCoords = $state<ComponentCoords | null>(null);
    let requestId = 0;

    function load(layer: string, cIdx: number) {
        currentCoords = { layer, cIdx };
        const thisRequestId = ++requestId;
        const componentKey = `${layer}:${cIdx}`;

        // Set loading states
        componentDetail = { status: "loading" };
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };
        datasetAttributions = { status: "loading" };
        interpretationDetail = { status: "loading" };

        const isStale = () => requestId !== thisRequestId;

        const startTime = performance.now();

        getComponentDataBundle(layer, cIdx, CORRELATIONS_TOP_K, TOKEN_STATS_TOP_K, DATASET_ATTRIBUTIONS_TOP_K, DETAIL_LIMIT)
            .then((bundle) => {
                if (isStale()) return;

                const elapsed = performance.now() - startTime;
                if (PERF_TIMING_ENABLED) {
                    const errorKeys = Object.keys(bundle.errors);
                    console.log(
                        `[ComponentDataBundled ${componentKey}] Loaded in ${elapsed.toFixed(0)}ms`,
                        errorKeys.length > 0 ? `(errors: ${errorKeys.join(", ")})` : "",
                    );
                }

                // Unpack bundle into individual state
                if (bundle.component_detail) {
                    componentDetail = { status: "loaded", data: bundle.component_detail };
                } else if (bundle.errors.component_detail) {
                    componentDetail = { status: "error", error: new Error(bundle.errors.component_detail) };
                } else {
                    componentDetail = { status: "error", error: new Error("No component detail in bundle") };
                }

                if (bundle.correlations !== null) {
                    correlations = { status: "loaded", data: bundle.correlations };
                } else if (bundle.errors.correlations) {
                    // 404 = no data for this component (treat as null, not error)
                    if (bundle.errors.correlations.includes("404") || bundle.errors.correlations.includes("not found")) {
                        correlations = { status: "loaded", data: null };
                    } else {
                        correlations = { status: "error", error: new Error(bundle.errors.correlations) };
                    }
                } else {
                    correlations = { status: "loaded", data: null };
                }

                if (bundle.token_stats !== null) {
                    tokenStats = { status: "loaded", data: bundle.token_stats };
                } else if (bundle.errors.token_stats) {
                    if (bundle.errors.token_stats.includes("404") || bundle.errors.token_stats.includes("not found")) {
                        tokenStats = { status: "loaded", data: null };
                    } else {
                        tokenStats = { status: "error", error: new Error(bundle.errors.token_stats) };
                    }
                } else {
                    tokenStats = { status: "loaded", data: null };
                }

                if (bundle.attributions !== null) {
                    datasetAttributions = { status: "loaded", data: bundle.attributions };
                } else if (bundle.errors.attributions) {
                    if (bundle.errors.attributions.includes("404") || bundle.errors.attributions.includes("not found")) {
                        datasetAttributions = { status: "loaded", data: null };
                    } else {
                        datasetAttributions = { status: "error", error: new Error(bundle.errors.attributions) };
                    }
                } else {
                    datasetAttributions = { status: "loaded", data: null };
                }

                if (bundle.interpretation_detail !== null) {
                    interpretationDetail = { status: "loaded", data: bundle.interpretation_detail };
                } else if (bundle.errors.interpretation_detail) {
                    if (
                        bundle.errors.interpretation_detail.includes("404") ||
                        bundle.errors.interpretation_detail.includes("not found")
                    ) {
                        interpretationDetail = { status: "loaded", data: null };
                    } else {
                        interpretationDetail = { status: "error", error: new Error(bundle.errors.interpretation_detail) };
                    }
                } else {
                    interpretationDetail = { status: "loaded", data: null };
                }
            })
            .catch((error) => {
                if (isStale()) return;

                const elapsed = performance.now() - startTime;
                if (PERF_TIMING_ENABLED) {
                    console.error(`[ComponentDataBundled ${componentKey}] Failed after ${elapsed.toFixed(0)}ms:`, error);
                }

                // Set all to error state
                componentDetail = { status: "error", error };
                correlations = { status: "error", error };
                tokenStats = { status: "error", error };
                datasetAttributions = { status: "error", error };
                interpretationDetail = { status: "error", error };
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
