

# general

- [x] **[BUG]** fix issue with broken histograms
- [x] embeddings view
- [x] a view where we can see, for some selected components, their activations (both actual and DT predicted) on a random set of samples -- stacked vertically so that we can compare. kinda like https://attention-motifs.github.io/v1/vis/attnpedia/index.html
- [ ] not just current token, but true next token and predicted next token when a given component is active
- [ ] proper test vs train split for decision trees

## performance

- [ ] parallelize decision tree computation??
  - memory will be the likely bottleneck here unless we can figure out how to make it read-only shareable
- [ ] re-compress big files
  - i.e. any given npy or jsonl file is originally compressed because it's in the zanj file, but we extract
  - hence, zanj.js should be rewritten to check for `{fname}.zip` first before loading `{fname}`

# interfaces

## index.html

- [ ] link to wandb run
- [ ] columns with measures of token entropy
- [ ] fix table csv export

## component.html

- [ ] proportion of all tokens that is any given token in table
- [ ] dists of activations in each sample
- [ ] hovering a token highlights all occurrences in the text
- [ ] token statistics -- highlight percentages by magnitude
- [ ] dists of activations per token?

## trees.html

- [ ] links to component/tree view for every component in view
- [ ] also show same-layer components similar to target component
  - [ ] column for "type" -- i.e. target, same layer, decision tree node
- [ ] max activating P(active|token) and P(token|active) tokens for each component? or at least target component
- [ ] ability to expand any of the nodes into its own tree
- [ ] no separate "leaf" nodes the way we have them now. certain nodes are leaves. Green if node active -> target active, red if node active -> target inactive
- [ ] ability to reorder rows
  - [ ] initial ordering should be by depth in tree
    - [ ] label direction of dependency, right now you have to infer
- [ ] in table include info on:
  - [ ] how accurate this node's own decision tree is
  - [ ] if branch, percentage of samples that reach here, go left, go right
  - [ ] if leaf, whether leaf predicts active or inactive for target component, and with what probability it's correct
- [ ] hovering a token should highlight that token in all rows for this column
- [ ] show distribution of the actual predictions for each token!!
  - [ ] maybe top few logits even?
