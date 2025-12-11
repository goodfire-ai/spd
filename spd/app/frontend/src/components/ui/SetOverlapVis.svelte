<script lang="ts">
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

    const countAOnly = $derived(Math.max(0, countA - countIntersection));
    const countBOnly = $derived(Math.max(0, countB - countIntersection));

    // Cumulative widths (all starting from x=0, stacked back to front)
    // Back layer: total (white)
    // Then: A-only + intersection + B-only (red for B-only visible portion)
    // Then: A-only + intersection (purple for intersection visible portion)
    // Front: A-only (blue)
    const pctTotal = 100;
    const pctUnion = $derived(((countAOnly + countIntersection + countBOnly) / totalCount) * 100);
    const pctAWithIntersection = $derived(((countAOnly + countIntersection) / totalCount) * 100);
    const pctAOnly = $derived((countAOnly / totalCount) * 100);
</script>

<div class="set-overlap-vis" title="A: {countA}, B: {countB}, A∩B: {countIntersection}">
    <!-- Back to front: total (white) -> union shows B-only (red) -> A+intersection (purple) -> A-only (blue) -->
    <div class="bar leftover" style="width: {pctTotal}%"></div>
    <div class="bar other" style="width: {pctUnion}%"></div>
    <div class="bar both" style="width: {pctAWithIntersection}%"></div>
    <div class="bar self" style="width: {pctAOnly}%"></div>
</div>

<style>
    .set-overlap-vis {
        position: relative;
        width: 200px;
        height: 4px;
    }

    .bar {
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
    }

    .leftover {
        background: white;
    }

    .other {
        background: rgb(255, 0, 0);
    }

    .both {
        background: rgb(176, 0, 176);
    }

    .self {
        background: rgb(0, 0, 255);
    }
</style>
