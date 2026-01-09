import { getContext } from "svelte";
import type { Loadable } from ".";
import {
    ApiError,
    getComponentCorrelations,
    getComponentTokenStats,
    getInterpretationPrompt,
    requestComponentInterpretation,
} from "./api";
import type { Interpretation } from "./api";
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

/** Prompt state: not requested, loading, loaded, or error */
export type PromptState =
    | { status: "none" }
    | { status: "loading" }
    | { status: "loaded"; data: string }
    | { status: "error"; error: unknown };

/**
 * Fetches all data for a component: detail, correlations, token stats.
 * Interpretation is derived from the global runState cache.
 * Handles stale request cancellation when coords change.
 */
export function useComponentData(getCoords: () => ComponentCoords | null) {
    const runState = getContext<RunStateContext>(RUN_STATE_KEY);
    let componentDetail = $state<Loadable<ComponentDetail>>({ status: "uninitialized" });
    // null inside Loadable means "no data for this component" (404)
    let correlations = $state<Loadable<ComponentCorrelations | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStats | null>>({ status: "uninitialized" });

    // Track which component is currently being generated (local to this hook instance)
    let generatingFor = $state<string | null>(null);
    let generationError = $state<{ key: string; error: unknown } | null>(null);

    // Prompt state (fetched alongside other component data)
    let prompt = $state<PromptState>({ status: "none" });

    // Effect for fetching componentDetail, correlations, tokenStats, prompt
    // Only re-runs when coords change
    $effect(() => {
        const coords = getCoords();
        if (!coords) {
            componentDetail = { status: "uninitialized" };
            correlations = { status: "uninitialized" };
            tokenStats = { status: "uninitialized" };
            prompt = { status: "none" };
            return;
        }

        const { layer, cIdx } = coords;
        let stale = false;

        // Set loading state
        componentDetail = { status: "loading" };
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };
        prompt = { status: "loading" };

        // Fetch component detail (cached in runState after first call)
        runState
            .getComponentDetail(layer, cIdx)
            .then((data) => {
                if (stale) return;
                componentDetail = { status: "loaded", data };
            })
            .catch((error) => {
                if (stale) return;
                componentDetail = { status: "error", error };
            });

        // Fetch correlations (404 = no data for this component)
        getComponentCorrelations(layer, cIdx, CORRELATIONS_TOP_K)
            .then((data) => {
                if (stale) return;
                correlations = { status: "loaded", data };
            })
            .catch((error) => {
                if (stale) return;
                if (error instanceof ApiError && error.status === 404) {
                    correlations = { status: "loaded", data: null };
                } else {
                    correlations = { status: "error", error };
                }
            });

        // Fetch token stats (404 = no data for this component)
        getComponentTokenStats(layer, cIdx, TOKEN_STATS_TOP_K)
            .then((data) => {
                if (stale) return;
                tokenStats = { status: "loaded", data };
            })
            .catch((error) => {
                if (stale) return;
                if (error instanceof ApiError && error.status === 404) {
                    tokenStats = { status: "loaded", data: null };
                } else {
                    tokenStats = { status: "error", error };
                }
            });

        // Fetch prompt (404 = no interpretation for this component)
        getInterpretationPrompt(layer, cIdx)
            .then((data) => {
                if (stale) return;
                prompt = { status: "loaded", data };
            })
            .catch((error) => {
                if (stale) return;
                if (error instanceof ApiError && error.status === 404) {
                    prompt = { status: "none" };
                } else {
                    prompt = { status: "error", error };
                }
            });

        return () => {
            stale = true;
        };
    });

    // Interpretation is derived from the global cache - reactive to both coords and cache
    const interpretation = $derived.by((): InterpretationState => {
        const coords = getCoords();
        if (!coords) return { status: "none" };

        const componentKey = `${coords.layer}:${coords.cIdx}`;

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
        const coords = getCoords();
        if (!coords || generatingFor !== null) return;

        const { layer, cIdx } = coords;
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
        get prompt() {
            return prompt;
        },
        generateInterpretation,
    };
}
