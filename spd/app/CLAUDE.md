# App TODO

## LocalAttributionsGraph

- [ ] Test edge highlighting via SVG string vs DOM manipulation
  - Current approach uses `{@html}` for bulk edge rendering (performance), but requires DOM manipulation for highlighting
  - Alternative: Re-render edges reactively as Svelte components
  - Need to benchmark both approaches with large graphs (10k+ edges)
