/**
 * Global run-scoped state store
 *
 * Holds state that is tied to the currently loaded run and accessed
 * throughout the component tree. Using a global store eliminates
 * prop drilling for these commonly-needed values.
 */

import type { Loadable } from ".";
import * as attrApi from "./localAttributionsApi";
import type { Interpretation } from "./localAttributionsApi";
import type { ComponentDetail } from "./localAttributionsTypes";

class RunState {
    /** Interpretation labels keyed by component key (layer:cIdx) */
    interpretations = $state<Record<string, Interpretation>>({});

    /** Lazy-loaded component details keyed by component key (layer:cIdx) */
    componentDetails = $state<Record<string, Loadable<ComponentDetail>>>({});

    /** Load all interpretations from the server */
    async loadInterpretations() {
        this.interpretations = await attrApi.getAllInterpretations();
    }

    /** Add a newly generated interpretation to the cache */
    addInterpretation(componentKey: string, interp: Interpretation) {
        this.interpretations = { ...this.interpretations, [componentKey]: interp };
    }

    /** Get interpretation for a component, if available */
    getInterpretation(componentKey: string): Interpretation | undefined {
        return this.interpretations[componentKey];
    }

    /** Load component detail (lazy, with caching) */
    async loadComponentDetail(layer: string, cIdx: number) {
        const cacheKey = `${layer}:${cIdx}`;
        if (this.componentDetails[cacheKey]?.status === "loading") return;

        this.componentDetails[cacheKey] = { status: "loading" };
        try {
            const detail = await attrApi.getComponentDetail(layer, cIdx);
            this.componentDetails[cacheKey] = { status: "loaded", data: detail };
        } catch (error) {
            this.componentDetails[cacheKey] = { status: "error", error };
        }
    }

    /** Get component detail from cache */
    getComponentDetail(layer: string, cIdx: number): Loadable<ComponentDetail> {
        const cacheKey = `${layer}:${cIdx}`;
        return this.componentDetails[cacheKey] ?? null;
    }

    /** Clear all run-scoped state (call when run changes or unloads) */
    clear() {
        this.interpretations = {};
        this.componentDetails = {};
    }
}

export const runState = new RunState();
