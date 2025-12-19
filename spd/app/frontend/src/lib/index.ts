/**
 * A type that represents a value that may be loading, loaded, or in an error state.
 * This is useful for handling asynchronous data in a type-safe way.
 *
 * The reason to have a null state is to allow for initial state before the loading starts.
 * This is distinct from having `T = null` which would also be valid, but is semantically different.
 */
export type Loadable<T> =
    | null // uninitialized
    | { status: "loading" }
    | { status: "loaded"; data: T }
    | { status: "error"; error: unknown };
