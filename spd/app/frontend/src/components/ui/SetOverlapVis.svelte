<script lang="ts">
    /**
     * Visualizes the overlap between two sets (subject A and object B).
     *
     * Layout:
     * - Both bars start at x=0, overlaid on each other
     * - Overlap region appears darker due to combined opacity
     * - Total width represents totalCount (e.g. total tokens in dataset)
     */

    type Props = {
        /** Number of items in set A (subject/query component) */
        countA: number;
        /** Number of items in set B (object/other component) */
        countB: number;
        /** Number of items in intersection A ∩ B */
        countIntersection: number;
        /** Total count to scale against (e.g. total tokens in dataset) */
        totalCount: number;
    };

    let { countA, countB, countIntersection, totalCount }: Props = $props();

    // Percentages for layout (relative to totalCount)
    if (totalCount <= 0) throw new Error(`totalCount must be positive, got ${totalCount}`);
    const countAOnly = $derived(Math.max(0, countA - countIntersection));
    const countBOnly = $derived(Math.max(0, countB - countIntersection));

    const pctA = $derived((countA / totalCount) * 100);
    const pctAOnly = $derived((countAOnly / totalCount) * 100);
    const pctIntersection = $derived((countIntersection / totalCount) * 100);
    const pctBOnly = $derived((countBOnly / totalCount) * 100);
    const barHeight = 4;
    /**
     *
     * Total:        |----------------------------------|
     * A:            |-------|
     * B:                 |-------------|
     *
     * For coloring:
     * AOnly:        |----| (blue)
     * Intersection:      |--| (purple)
     * BOnly:                |----------| (red)
     */
</script>

<div class="set-overlap-vis" title="A: {countA}, B: {countB}, A∩B: {countIntersection}">
    <div class="bars-container" style="height: {barHeight}px">
        <div class="bar self" style="width: {pctAOnly}%"></div>
        <div class="bar both" style="width: {pctIntersection}%; left: {pctAOnly}%"></div>
        <div class="bar other" style="width: {pctBOnly}%; left: {pctA}%"></div>
    </div>
</div>

<style>
    .set-overlap-vis {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        min-width: 60px;
    }

    .bars-container {
        position: relative;
        width: 100%;
        background: var(--bg-inset);
    }

    .bar {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        min-width: 1px;
    }

    .self {
        background: blue;
    }

    .both {
        background: rgb(176, 0, 176);
    }

    .other {
        background: red;
    }
</style>
