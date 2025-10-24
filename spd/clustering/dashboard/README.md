# 2025-10-02 15:17

For cluster view:

- SAE/logitlens/tunedlens decoding of read and write directions for each cluster
- github comments interface
- more plots
- base frequency of each token in dataset
- do some kind of clustering on tokens via embedding space, to label their groups. ideally so we can make a histogram of which groups of tokens this cluster activates on
- measure of how "concentrated" the subcomponents are per module
- measure of depth in the model
- measure of how "attention-y" vs "MLP-y" the clusters subcomponents are
- skewed vs uniform activation frequencies, skewed vs uniform max act position
    - maybe not max act pos, but overall mass across positions

For list view (or wandb view?):

- stats of tok concentration, entropy, subcomp in module concentration, etc across all clusters
- some kind of embedding of clusters, 3d view, click on pt to go to cluster view





# TODO

## static:


- makeup of activations on a per-subcomponent basis
- makeup of clusters -- which modules do they have subcomponents from?
- [ ] features in 2d plane -- display them as vector fields, with points in that 2d plane colored corresponding to various SAE features





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
