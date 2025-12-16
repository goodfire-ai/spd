// Placeholder - exports currently handled in individual modules

export type Loadable<T> =
    | null
    | { status: "loading" }
    | { status: "loaded"; data: T }
    | { status: "error"; error: unknown };
