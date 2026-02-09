/**
 * Shared graph layout utilities for attribution visualizations.
 *
 * Replaces hardcoded parseLayer/ROW_ORDER/QKV_SUBTYPES with generic
 * layout driven by ModelInfo from the backend.
 */

import type { ModelInfo } from "./api/runs";

export type LayerInfo = {
    name: string;
    block: number; // -1 for embed, Infinity for output, else block index
    role: string; // last segment of dotted path, or "wte"/"output" for pseudo-layers
    group: string | null; // group key if this role belongs to a role_group, else null
};

/**
 * Parse a layer name into structured info.
 *
 * Works with any dotted module path: extracts the first integer as block,
 * last segment as role. Special-cases "wte" and "output" pseudo-layers.
 */
export function parseLayer(name: string, modelInfo: ModelInfo): LayerInfo {
    if (name === "wte") {
        return { name, block: -1, role: "wte", group: null };
    }
    if (name === "output") {
        return { name, block: Infinity, role: "output", group: null };
    }

    // Extract block index: first integer segment in dotted path
    const segments = name.split(".");
    let block = Infinity - 1; // fallback for paths without block (e.g. "lm_head")
    for (const seg of segments) {
        if (/^\d+$/.test(seg)) {
            block = +seg;
            break;
        }
    }

    // Role is the last segment
    const role = segments[segments.length - 1];

    // Check if this role belongs to a group
    let group: string | null = null;
    for (const [groupName, roles] of Object.entries(modelInfo.role_groups)) {
        if (roles.includes(role)) {
            group = groupName;
            break;
        }
    }

    return { name, block, role, group };
}

/**
 * Get the visual row key for a layer. Grouped roles share a row key.
 *
 * For grouped roles (e.g. QKV), the row key is "block.groupName" (e.g. "0.qkv").
 * For ungrouped roles, the row key is the layer name itself.
 */
export function getRowKey(layer: string, modelInfo: ModelInfo): string {
    const info = parseLayer(layer, modelInfo);
    if (info.group !== null) {
        return `${info.block}.${info.group}`;
    }
    return layer;
}

/**
 * Build the row ordering for sorting. Returns an ordered list of role keys
 * (with groups collapsed) that determines vertical position.
 *
 * Order: wte, then roles in execution order (groups collapsed), then output.
 */
export function buildRowOrder(modelInfo: ModelInfo): string[] {
    const order: string[] = ["wte"];
    const seen = new Set<string>();

    for (const role of modelInfo.role_order) {
        // Check if role belongs to a group
        let key = role;
        for (const [groupName, roles] of Object.entries(modelInfo.role_groups)) {
            if (roles.includes(role)) {
                key = groupName;
                break;
            }
        }
        if (!seen.has(key)) {
            seen.add(key);
            order.push(key);
        }
    }

    order.push("output");
    return order;
}

/**
 * Get display label for a row.
 */
export function getRowLabel(layer: string, modelInfo: ModelInfo): string {
    const info = parseLayer(layer, modelInfo);
    if (layer === "wte" || layer === "output") return layer;

    // Check display_names for the role
    const displayName = modelInfo.display_names[info.role];
    if (displayName) return displayName;

    // For grouped roles, show "block.group" (e.g. "0.q/k/v")
    if (info.group !== null) {
        const groupRoles = modelInfo.role_groups[info.group];
        const shortNames = groupRoles.map((r) => r.replace(/_proj$/, ""));
        return `${info.block}.${shortNames.join("/")}`;
    }

    return `${info.block}.${info.role}`;
}

/**
 * Sort row keys by block index, then by role order within a block.
 */
export function sortRows(rows: string[], modelInfo: ModelInfo): string[] {
    const rowOrder = buildRowOrder(modelInfo);

    const parseRow = (r: string) => {
        if (r === "wte") return { block: -1, role: "wte" };
        if (r === "output") return { block: Infinity, role: "output" };

        // Try "block.groupOrRole" format (e.g. "0.qkv" or "3.c_fc")
        const dotIdx = r.indexOf(".");
        if (dotIdx !== -1) {
            const blockStr = r.substring(0, dotIdx);
            if (/^\d+$/.test(blockStr)) {
                return { block: +blockStr, role: r.substring(dotIdx + 1) };
            }
        }

        // Fallback: parse as a layer name
        const info = parseLayer(r, modelInfo);
        return { block: info.block, role: info.group ?? info.role };
    };

    return [...rows].sort((a, b) => {
        const infoA = parseRow(a);
        const infoB = parseRow(b);
        if (infoA.block !== infoB.block) return infoA.block - infoB.block;
        const idxA = rowOrder.indexOf(infoA.role);
        const idxB = rowOrder.indexOf(infoB.role);
        return idxA - idxB;
    });
}

/**
 * Get the grouped roles (e.g. QKV subtypes) for a given group name, if any.
 * Returns null if no group exists.
 */
export function getGroupRoles(groupName: string, modelInfo: ModelInfo): string[] | null {
    return modelInfo.role_groups[groupName] ?? null;
}

/**
 * Reconstruct the full layer path for a role within a specific block.
 *
 * Looks up the module_paths to find a matching path with the given block and role.
 * Returns null if no match is found.
 */
export function getLayerPath(block: number, role: string, modelInfo: ModelInfo): string | null {
    for (const path of modelInfo.module_paths) {
        const segments = path.split(".");
        const pathRole = segments[segments.length - 1];
        if (pathRole !== role) continue;

        for (const seg of segments) {
            if (/^\d+$/.test(seg) && +seg === block) {
                return path;
            }
        }
    }
    return null;
}
