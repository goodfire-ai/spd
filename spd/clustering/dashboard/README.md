# TODO

## static:


- makeup of activations on a per-subcomponent basis
- makeup of clusters -- which modules do they have subcomponents from?
- [ ] features in 2d plane -- display them as vector fields, with points in that 2d plane colored corresponding to various SAE features


## interactive:

minimal example: put in a piece of text, it computes cluster activations on it



# causal masks

one of the things we might want to do is:

- define a causal mask, using:
	- some subset of the data (a set of prompts)
	- some other method
- run inference using that particular causal mask on some other dataset

this requires an easy way to define and use custom causal masks. a good solution might be something like:

- interface to define a causal mask, by providing a dataset and/or manually editing
	- it should have a button to "label" a causal mask -- probably, we can hash the causal mask, save the mask to a file on the backend, and use that hash as a key
	- copy the hash
- in other interfaces for doing inference with the mask, we can paste the hash to specify a causal mask

# Enhanced Cluster Dashboard

## Current State Analysis

The current index.html has:
- Basic HTML table with manual sorting via onclick handlers
- Custom cluster-selection.js with processTableData(), sortTableData(), renderTable()
- Model info display panel
- Basic statistics columns (Component count, samples, activation stats)
- Link to detail view (cluster.html)

## Planned Refactor

### New Table Structure with util/table.js
Replace the manual table implementation with DataTable class:
- Automatic sorting/filtering on all columns
- Pagination for large datasets
- CSV export functionality
- Consistent styling and UX

### Enhanced Columns
1. **Cluster ID** - sortable, filterable
2. **Model View** - inline visualization using model-visualization.js
   - Compact heatmap showing component distribution across model architecture
   - Tooltip on hover for module details
3. **Component Count** - numeric with filtering (>50, <100, etc.)
4. **Module Distribution** - text summary of which modules contain components
5. **Sparklines** - activation distribution visualizations
   - Max/Mean/Median activation sparklines over samples
   - Compact SVG charts using util/sparklines.js
6. **Sample Count** - numeric with filtering
7. **Activation Stats** - formatted numeric values
8. **Actions** - links to detail views

### Technical Implementation Plan

#### Data Processing Pipeline
1. Load cluster data and model info (existing)
2. Process raw cluster data into table-friendly format:
   - Extract component statistics per cluster
   - Prepare activation arrays for sparklines
   - Generate model architecture data for each cluster
3. Configure DataTable with custom renderers for complex columns

#### Custom Column Renderers
- **Model View**: Call renderModelArchitecture() and renderToHTML(), return compact version
- **Sparklines**: Generate SVG using sparkline() function with activation data
- **Module Distribution**: Smart text formatting with tooltips for full lists

#### Integration Points
- Reuse model-visualization.js utilities (pure functions)
- Leverage util/sparklines.js for activation distributions
- Maintain existing data loading from max_activations_*.json and model_info.json

### Benefits
- Better UX with proper table functionality (sort, filter, paginate)
- Rich visualizations directly in the overview table
- Faster cluster comparison without clicking into details
- Consistent styling and behavior patterns
- Maintainable code using utility modules

### File Changes Required
- index.html: Major restructure to use DataTable
- cluster-selection.js: Refactor to data processing + DataTable setup
- New custom renderers for model view and sparkline columns
- CSS updates for embedded visualizations





