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





