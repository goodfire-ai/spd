import { getComponentInterpretation, requestComponentInterpretation, type Interpretation } from "./api";
import { runState } from "./runState.svelte";

/** Interpretation can be: none, loading, generating, loaded, or error */
export type InterpretationState =
    | { status: "none" }
    | { status: "loading" }
    | { status: "generating" }
    | { status: "loaded"; data: Interpretation }
    | { status: "error"; error: unknown };

/**
 * Composable for managing interpretation state for a component.
 * Fetches existing interpretation on mount/change, and provides a request function for generating new ones.
 */
export function useInterpretation(getCoords: () => { layer: string; cIdx: number } | null) {
    let interpretation = $state<InterpretationState>({ status: "none" });

    $effect(() => {
        const coords = getCoords();
        if (!coords) {
            interpretation = { status: "none" };
            return;
        }
        const { layer, cIdx } = coords;
        let stale = false;
        interpretation = { status: "loading" };

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
        get interpretation() {
            return interpretation;
        },
        generateInterpretation,
    };
}
