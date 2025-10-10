<script lang="ts">
    export let items: any[] = [];
    export let rowHeight = 200;
    export let buffer = 6;
    export let getKey: (item: any, index: number) => string | number = (_item, index) => index;

    let viewportHeight = 0;
    let scrollTop = 0;

    $: totalHeight = items.length * rowHeight;
    $: startIndex = Math.max(0, Math.floor(scrollTop / rowHeight) - buffer);
    $: endIndex = Math.min(
        items.length,
        Math.ceil((scrollTop + viewportHeight) / rowHeight) + buffer,
    );
    $: visible = items.slice(startIndex, endIndex);
</script>

<div class="virtual-list">
    <div
        class="virtual-viewport"
        bind:clientHeight={viewportHeight}
        on:scroll={(event) => (scrollTop = event.currentTarget.scrollTop)}
    >
        <div class="virtual-spacer" style={`height:${totalHeight}px`}>
            <div
                class="virtual-content"
                style={`transform: translateY(${startIndex * rowHeight}px);`}
            >
                {#each visible as item, localIndex (getKey(item, startIndex + localIndex))}
                    <slot
                        {item}
                        index={startIndex + localIndex}
                    ></slot>
                {/each}
            </div>
        </div>
    </div>
</div>

<style>
    .virtual-list {
        position: relative;
        width: 100%;
    }

    .virtual-viewport {
        overflow-y: auto;
        position: relative;
        width: 100%;
    }

    .virtual-spacer {
        position: relative;
        width: 100%;
    }

    .virtual-content {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        will-change: transform;
    }
</style>
