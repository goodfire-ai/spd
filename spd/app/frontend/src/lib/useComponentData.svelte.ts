import { getContext } from "svelte";
import type { Loadable } from ".";
import {
    ApiError,
    getComponentCorrelations,
    getComponentTokenStats,
    getInterpretationDetail,
    requestComponentInterpretation,
} from "./api";
import type { Interpretation, InterpretationDetail } from "./api";
import type { ComponentCorrelations, ComponentDetail, TokenStats } from "./localAttributionsTypes";
import { RUN_STATE_KEY, type RunStateContext } from "./runState.svelte";

/** Correlations are paginated in the UI, so fetch more */
const CORRELATIONS_TOP_K = 100;
/** Token stats are displayed directly (max 50 shown) */
const TOKEN_STATS_TOP_K = 50;

export type ComponentCoords = { layer: string; cIdx: number };

/** Interpretation can be: none, loading, generating, loaded, or error */
export type InterpretationState =
    | { status: "none" }
    | { status: "loading" }
    | { status: "generating" }
    | { status: "loaded"; data: Interpretation }
    | { status: "error"; error: unknown };


/**
 * Hook for loading component data (detail, correlations, token stats, interpretation detail).
 *
 * Call `load(layer, cIdx)` explicitly when you want to fetch data.
 * Interpretation headline is derived from the global runState cache.
 * Interpretation detail (reasoning + prompt) is fetched on-demand.
 */
export function useComponentData() {
    const runState = getContext<RunStateContext>(RUN_STATE_KEY);

    let componentDetail = $state<Loadable<ComponentDetail>>({ status: "uninitialized" });
    // null inside Loadable means "no data for this component" (404)
    let correlations = $state<Loadable<ComponentCorrelations | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStats | null>>({ status: "uninitialized" });
    let interpretationDetail = $state<Loadable<InterpretationDetail | null>>({ status: "uninitialized" });

    // Current coords being loaded/displayed (for interpretation lookup)
    let currentCoords = $state<ComponentCoords | null>(null);

    // Track which component is currently being generated (local to this hook instance)
    let generatingFor = $state<string | null>(null);
    let generationError = $state<{ key: string; error: unknown } | null>(null);

    // Request counter for handling stale responses
    let requestId = 0;

    /**
     * Load all data for the given component.
     * Call this from event handlers or on mount.
     */
    function load(layer: string, cIdx: number) {
        currentCoords = { layer, cIdx };
        const thisRequestId = ++requestId;

        // Set loading states
        componentDetail = { status: "loading" };
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };
        interpretationDetail = { status: "loading" };

        // Clear any previous generation error for different component
        const componentKey = `${layer}:${cIdx}`;
        if (generationError?.key !== componentKey) {
            generationError = null;
        }

        // Helper to check if this request is still current
        const isStale = () => requestId !== thisRequestId;

        // Fetch component detail (cached in runState after first call)
        runState
            .getComponentDetail(layer, cIdx)
            .then((data) => {
                if (isStale()) return;
                componentDetail = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                componentDetail = { status: "error", error };
            });

        // Fetch correlations (404 = no data for this component)
        getComponentCorrelations(layer, cIdx, CORRELATIONS_TOP_K)
            .then((data) => {
                if (isStale()) return;
                correlations = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    correlations = { status: "loaded", data: null };
                } else {
                    correlations = { status: "error", error };
                }
            });

        // Fetch token stats (404 = no data for this component)
        getComponentTokenStats(layer, cIdx, TOKEN_STATS_TOP_K)
            .then((data) => {
                if (isStale()) return;
                tokenStats = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    tokenStats = { status: "loaded", data: null };
                } else {
                    tokenStats = { status: "error", error };
                }
            });

        // Fetch interpretation detail (404 = no interpretation for this component)
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

    /**
     * Reset all state to uninitialized.
     */
    function reset() {
        requestId++; // Invalidate any in-flight requests
        currentCoords = null;
        componentDetail = { status: "uninitialized" };
        correlations = { status: "uninitialized" };
        tokenStats = { status: "uninitialized" };
        interpretationDetail = { status: "uninitialized" };
    }

    // Interpretation is derived from the global cache - reactive to both coords and cache
    const interpretation = $derived.by((): InterpretationState => {
        if (!currentCoords) return { status: "none" };

        const componentKey = `${currentCoords.layer}:${currentCoords.cIdx}`;

        // Check if we're currently generating for this component
        if (generatingFor === componentKey) {
            return { status: "generating" };
        }

        // Check if there was a generation error for this component
        if (generationError?.key === componentKey) {
            return { status: "error", error: generationError.error };
        }

        // Show loading while global cache is loading
        if (runState.interpretations.status === "loading") return { status: "loading" };
        if (runState.interpretations.status !== "loaded") return { status: "none" };
        const cached = runState.interpretations.data[componentKey];
        return cached ? { status: "loaded", data: cached } : { status: "none" };
    });

    async function generateInterpretation() {
        if (!currentCoords || generatingFor !== null) return;

        const { layer, cIdx } = currentCoords;
        const componentKey = `${layer}:${cIdx}`;

        generatingFor = componentKey;
        generationError = null;

        try {
            const result = await requestComponentInterpretation(layer, cIdx);
            // Update the global cache - this will reactively update the derived interpretation
            runState.setInterpretation(componentKey, result);
        } catch (e) {
            generationError = { key: componentKey, error: e instanceof Error ? e.message : String(e) };
        } finally {
            generatingFor = null;
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
