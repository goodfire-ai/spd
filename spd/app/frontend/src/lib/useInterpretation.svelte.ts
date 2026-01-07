import type { Loadable } from ".";
import { getComponentInterpretation, requestComponentInterpretation, type Interpretation } from "./api";
import { runState } from "./runState.svelte";

/**
 * Composable for managing interpretation state for a component.
 * Fetches existing interpretation on mount/change, and provides a request function for generating new ones.
 */
export function useInterpretation(getCoords: () => { layer: string; cIdx: number } | null) {
    let interpretation = $state<Loadable<Interpretation>>(null);

    $effect(() => {
        const coords = getCoords();
        if (!coords) {
            interpretation = null;
            return;
        }
        const { layer, cIdx } = coords;

        interpretation = { status: "loading" };
        getComponentInterpretation(layer, cIdx)
            .then((data) => {
                interpretation = data ? { status: "loaded", data } : null;
            })
            .catch((error) => {
                interpretation = { status: "error", error };
            });
    });

    async function request() {
        const coords = getCoords();
        if (!coords || interpretation?.status === "loading") return;
        const { layer, cIdx } = coords;

        interpretation = { status: "loading" };
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
        request,
    };
}
