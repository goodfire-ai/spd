/**
 * Global run-scoped state store
 *
 * Holds state that is tied to the currently loaded run and accessed
 * throughout the component tree. Using a global store eliminates
 * prop drilling for these commonly-needed values.
 */

import { untrack } from "svelte";
import type { Loadable } from ".";
import * as api from "./api";
import type { RunState as RunData, Interpretation } from "./api";
import type { ComponentDetail } from "./localAttributionsTypes";

class RunState {
    /** The currently loaded run */
    run = $state<Loadable<RunData>>(null);

    /** Lazy-loaded interpretations keyed by component key (layer:cIdx) */
    interpretations = $state<Record<string, Loadable<Interpretation | null>>>({});

    /** Lazy-loaded component details keyed by component key (layer:cIdx) */
    componentDetails = $state<Record<string, Loadable<ComponentDetail>>>({});

    /** Load a run by wandb path */
    async loadRun(wandbPath: string, contextLength: number) {
        this.clear();
        this.run = { status: "loading" };
        try {
            await api.loadRun(wandbPath, contextLength);
            const status = await api.getStatus();
            if (status) {
                this.run = { status: "loaded", data: status };
            } else {
                this.run = { status: "error", error: "Failed to load run" };
            }
        } catch (error) {
            this.run = { status: "error", error };
        }
    }

    /** Check backend status and sync run state */
    async syncStatus() {
        try {
            const status = await api.getStatus();
            if (this.run?.status === "loaded" && this.run.data && !status) {
                this.run = { status: "error", error: "Backend state lost (restarted)" };
                return;
            }
            if (status) {
                this.run = { status: "loaded", data: status };
            } else {
                this.run = null;
            }
        } catch {
            if (this.run?.status === "loaded") {
                this.run = { status: "error", error: "Backend unreachable" };
            }
        }
    }

    /** Load interpretation (lazy, with caching) */
    async loadInterpretation(layer: string, cIdx: number) {
        const cacheKey = `${layer}:${cIdx}`;
        // Use untrack to avoid creating reactive dependency when checking cache
        const status = untrack(() => this.interpretations[cacheKey]?.status);
        if (status === "loading" || status === "loaded") return;

        this.interpretations[cacheKey] = { status: "loading" };
        try {
            const interp = await api.getComponentInterpretation(layer, cIdx);
            this.interpretations[cacheKey] = { status: "loaded", data: interp };
        } catch (error) {
            this.interpretations[cacheKey] = { status: "error", error };
        }
    }

    /** Get interpretation from cache */
    getInterpretation(layer: string, cIdx: number): Loadable<Interpretation | null> {
        const cacheKey = `${layer}:${cIdx}`;
        return this.interpretations[cacheKey] ?? null;
    }

    /** Load component detail (lazy, with caching) */
    async loadComponentDetail(layer: string, cIdx: number) {
        const cacheKey = `${layer}:${cIdx}`;
        if (this.componentDetails[cacheKey]?.status === "loading") return;

        this.componentDetails[cacheKey] = { status: "loading" };
        try {
            const detail = await api.getComponentDetail(layer, cIdx);
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

    /** Clear all run-scoped cached state (call when run changes) */
    clear() {
        this.interpretations = {};
        this.componentDetails = {};
    }

    /** Fully reset the store (including run) */
    reset() {
        this.run = null;
        this.clear();
    }
}

export const runState = new RunState();
