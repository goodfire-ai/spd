import type { Loadable } from ".";
import {
    getActivationContextsSummary,
    getComponentCorrelations,
    getComponentInterpretation,
    getComponentTokenStats,
    requestComponentInterpretation,
} from "./api";
import type { ComponentCorrelations, ComponentSummary, TokenStats } from "./localAttributionsTypes";
import { runState } from "./runState.svelte";

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

import type { Interpretation } from "./api";

/**
 * Fetches all data for a component: correlations, token stats, and interpretation.
 * Handles stale request cancellation when coords change.
 */
export function useComponentData(getCoords: () => ComponentCoords | null) {
    let componentSummary = $state<Loadable<ComponentSummary>>(null);

    // TODO why are these inner type nullable? this semantically conflicts with null as uninitialized
    let correlations = $state<Loadable<ComponentCorrelations | null>>(null);
    let tokenStats = $state<Loadable<TokenStats | null>>(null);

    let interpretation = $state<InterpretationState>({ status: "none" });

    $effect(() => {
        const coords = getCoords();
        if (!coords) {
            componentSummary = null;
            correlations = null;
            tokenStats = null;
            interpretation = { status: "none" };
            return;
        }

        const { layer, cIdx } = coords;
        let stale = false;

        // Set loading state
        componentSummary = { status: "loading" };
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };
        interpretation = { status: "loading" };

        // Fetch component summary
        getActivationContextsSummary()
            .then((data) => {
                if (stale) return;
                const summary = data[layer].find((s) => s.subcomponent_idx === cIdx);
                if (!summary) {
                    componentSummary = {
                        status: "error",
                        error: new Error(`Component summary not found for ${layer}:${cIdx}`),
                    };
                    return;
                }
                componentSummary = { status: "loaded", data: summary };
            })
            .catch((error) => {
                if (stale) return;
                componentSummary = { status: "error", error };
            });

        // Fetch correlations
        getComponentCorrelations(layer, cIdx, CORRELATIONS_TOP_K)
            .then((data) => {
                if (stale) return;
                correlations = { status: "loaded", data };
            })
            .catch((error) => {
                if (stale) return;
                correlations = { status: "error", error };
            });

        // Fetch token stats
        getComponentTokenStats(layer, cIdx, TOKEN_STATS_TOP_K)
            .then((data) => {
                if (stale) return;
                tokenStats = { status: "loaded", data };
            })
            .catch((error) => {
                if (stale) return;
                tokenStats = { status: "error", error };
            });

        // Fetch interpretation
        getComponentInterpretation(layer, cIdx)
            .then((data) => {
                if (stale) return;
                interpretation = data ? { status: "loaded", data } : { status: "none" };
            })
            .catch((error) => {
                if (stale) return;
                interpretation = { status: "error", error };
            });

        return () => {
            stale = true;
        };
    });

    async function generateInterpretation() {
        const coords = getCoords();
        if (!coords || interpretation?.status === "generating") return;
        const { layer, cIdx } = coords;

        interpretation = { status: "generating" };
        try {
            const result = await requestComponentInterpretation(layer, cIdx);
            interpretation = { status: "loaded", data: result };
            runState.loadInterpretations();
        } catch (e) {
            interpretation = { status: "error", error: e instanceof Error ? e.message : String(e) };
        }
    }

    return {
        get componentSummary() {
            return componentSummary;
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
        generateInterpretation,
    };
}
