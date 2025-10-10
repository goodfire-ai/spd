<script lang="ts">
    export let items: any[] = [];
    export let itemWidth = 600;
    export let buffer = 3;
    export let getKey: (item: any, index: number) => string | number = (_item, index) => index;

    let viewportWidth = 0;
    let scrollLeft = 0;

    $: totalWidth = items.length * itemWidth;
    $: startIndex = Math.max(0, Math.floor(scrollLeft / itemWidth) - buffer);
    $: endIndex = Math.min(
        items.length,
        Math.ceil((scrollLeft + (viewportWidth || itemWidth * 3)) / itemWidth) + buffer,
    );
    $: visible = items.slice(startIndex, endIndex);
    $: offsetLeft = startIndex * itemWidth;

    $: console.log('HVL:', { total: items.length, visible: visible.length, startIndex, endIndex });
</script>

<div class="asdf_container">
    <div
        class="scroller"
        bind:clientWidth={viewportWidth}
        on:scroll={(e) => (scrollLeft = e.currentTarget.scrollLeft)}
    >
        <div class="spacer" style:width="{totalWidth}px">
            <div class="content" style:margin-left="{offsetLeft}px">
                {#each visible as item, localIndex (getKey(item, startIndex + localIndex))}
                    <slot item={item} index={startIndex + localIndex}></slot>
                {/each}
            </div>
        </div>
    </div>
</div>

<style>
    .asdf_container {
        width: 100%;
        height: 100%;
    }

    .scroller {
        overflow-x: auto;
        height: 100%;
    }

    .spacer {
        margin: 1rem;
        height: 100%;
        position: relative;
    }

    .content {
        display: flex;
        gap: 1rem;
        height: fit-content;
    }
</style>
