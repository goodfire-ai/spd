/**
 * Global run-scoped state store
 *
 * Holds state that is tied to the currently loaded run and accessed
 * throughout the component tree. Using a global store eliminates
 * prop drilling for these commonly-needed values.
 */

import type { Loadable } from ".";
import * as api from "./api";
import type { RunState as RunData, Interpretation } from "./api";
import type { ActivationContextsSummary, ComponentDetail } from "./localAttributionsTypes";

class RunState {
    /** The currently loaded run */
    run = $state<Loadable<RunData>>(null);

    /** Interpretation labels keyed by component key (layer:cIdx) */
    interpretations = $state<Record<string, Interpretation>>({});

    /** Cached component details keyed by component key (layer:cIdx) - non-reactive */
    private _componentDetailsCache: Record<string, ComponentDetail> = {};

    /** Cached activation contexts summary (non-reactive to avoid dependency cycles) */
    private _summaryCache: ActivationContextsSummary | null = null;

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

    /** Load all interpretations from the server */
    async loadInterpretations() {
        this.interpretations = await api.getAllInterpretations();
    }

    /** Get interpretation for a component, if available */
    getInterpretation(componentKey: string): Interpretation | undefined {
        return this.interpretations[componentKey];
    }

    /** Get component detail (fetches once, then cached) */
    async getComponentDetail(layer: string, cIdx: number): Promise<ComponentDetail> {
        const cacheKey = `${layer}:${cIdx}`;
        if (cacheKey in this._componentDetailsCache) return this._componentDetailsCache[cacheKey];

        const detail = await api.getComponentDetail(layer, cIdx);
        this._componentDetailsCache[cacheKey] = detail;
        return detail;
    }

    /** Get activation contexts summary (fetches once, then cached) */
    async getActivationContextsSummary(): Promise<ActivationContextsSummary> {
        if (this._summaryCache) return this._summaryCache;

        const summary = await api.getActivationContextsSummary();
        this._summaryCache = summary;
        return summary;
    }

    /** Clear all run-scoped cached state (call when run changes) */
    clear() {
        this.interpretations = {};
        this._componentDetailsCache = {};
        this._summaryCache = null;
    }

    /** Fully reset the store (including run) */
    reset() {
        this.run = null;
        this.clear();
    }
}

export const runState = new RunState();
