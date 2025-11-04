# general

- [x] **[BUG]** fix issue with broken histograms
- [ ] not just current token, but true next token and predicted next token when a given component is active
- [ ] embeddings view
- [ ] a view where we can see, for some selected components, their activations (both actual and DT predicted) on a random set of samples -- stacked vertically so that we can compare. kinda like https://attention-motifs.github.io/v1/vis/attnpedia/index.html

# index.html

- [ ] link to wandb run
- [ ] columns with measures of token entropy
- [ ] fix table csv export


# component.html

- [ ] proportion of all tokens that is any given token in table
- [ ] dists of activations in each sample
- [ ] hovering a token highlights all occurrences in the text
- [ ] token statistics -- highlight percentages by magnitude
- [ ] dists of activations per token?