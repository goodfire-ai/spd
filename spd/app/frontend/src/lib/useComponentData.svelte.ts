import type { Loadable } from ".";
import {
    getComponentCorrelations,
    getComponentInterpretation,
    getComponentTokenStats,
    requestComponentInterpretation,
} from "./api";
import type { ComponentCorrelations, TokenStats } from "./localAttributionsTypes";
import { runState } from "./runState.svelte";
import { type InterpretationState } from "./useInterpretation.svelte";

export type ComponentCoords = { layer: string; cIdx: number };

export type { InterpretationState };

/**
 * Fetches all data for a component: correlations, token stats, and interpretation.
 * Handles stale request cancellation when coords change.
 */
export function useComponentData(getCoords: () => ComponentCoords | null) {
    let correlations = $state<Loadable<ComponentCorrelations | null>>(null);
    let tokenStats = $state<Loadable<TokenStats | null>>(null);
    let interpretation = $state<InterpretationState>({ status: "none" });

    $effect(() => {
        const coords = getCoords();
        if (!coords) {
            correlations = null;
            tokenStats = null;
            interpretation = { status: "none" };
            return;
        }

        const { layer, cIdx } = coords;
        let stale = false;

        // Set loading state
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };
        interpretation = { status: "loading" };

        // Fetch correlations
        getComponentCorrelations(layer, cIdx, 1000)
            .then((data) => {
                if (stale) return;
                correlations = { status: "loaded", data };
            })
            .catch((error) => {
                if (stale) return;
                correlations = { status: "error", error };
            });

        // Fetch token stats
        getComponentTokenStats(layer, cIdx, 1000)
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
