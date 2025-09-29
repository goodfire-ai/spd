# Design doc

We have several interfaces:

- streamlit subcomponent dashboard
- cluster selection interface (big table) [`index.html`](index.html)
- cluster individual view [`cluster.html`](cluster.html)
- inference with causal masks

rather than trying to put these all into one interface, we should instead:

- ensure that we can specify via URL various parameters specifying which cluster/component/causal mask/sample/etc we are looking at
   - using cookies/local storage/something server side to do this adds more complexity, makes it less observable to the user, and makes sharing specific views harder
- link between these interfaces using these URLs in some reasonable way

## Interface Flow Diagram

```mermaid
flowchart TD
    A[Streamlit Dashboard<br/>Subcomponent View] -->|"URL: ?component=X"| B[Cluster Selection<br/>index.html<br/>Big Table]
    B -->|"URL: ?cluster=Y"| C[Cluster Individual View<br/>cluster.html]
    C -->|"URL: ?mask_hash=Z"| D[Causal Mask Inference]

    B -->|"filter by component"| B
    C -->|"select samples"| E[Sample Viewer]

    D -->|"apply mask"| F[Inference Results]
    F -->|"view cluster details"| C

    A -->|"direct cluster link<br/>URL: ?cluster=Y"| C

    G[Causal Mask Editor] -->|"save mask<br/>returns hash"| H[Mask Storage<br/>Backend]
    H -->|"load mask<br/>by hash"| D

    E -->|"navigate back"| C
    C -->|"navigate back"| B

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style G fill:#fce4ec
```


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
