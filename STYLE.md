# Code Style Guide

TLDR:
- prioritise simple, straightforward code. Our users are researchers, often with little coding experience.
- safety: use types, einops, jaxtyping, and liberal assertions.
- fail fast - if something is wrong, the code should fail, not recover silently.


## Design / Architecture

We want to decouple metrics and analysis from the core codebase as much as possible. Metric- and figure-generation code should be encapsulated in `spd/metrics.py` and `spd/figures.py`.

### Fail Fast (Negative Space Programming)
Code should fail immediately when assumptions are violated, preventing bugs from propagating.

If there's an assumption you're making while writing code, assert it.
- If you were right, then it won't matter
- If you were wrong, then the code **should** fail.

## Type Annotations
- **Always** use jaxtyping for tensor or numpy array shapes (though for now we don't do runtime checking)
- Always use the PEP 604 typing format of `|` for unions and `type | None` over `Optional`.
- Use `dict`, `list` and `tuple` not `Dict`, `List` and `Tuple`
- Don't add type annotations only when they're redundant.
	- i.e. `my_thing: Thing = Thing()` or `name: str = "John Doe"` don't need type annotations.
	- however, `var = foo()` or `result = thing.bar()` should be annotated!
- FOR CLAUDE: don't worry about cleaning up unused imports, we do this automatically with ruff using `make format`

## Tensor Operations
- Try to use einops by default for clarity.
- Assert shapes liberally
- Document complex tensor manipulations

## Comments

Your first instinct should be: "If I couldn't write any comments, how would I write this code?"

**Don't**: Write Obvious Comments
**Do**: Write comments for complex logic

**Bad:**
```python
# get dataloader
dataloader = get_dataloader(config)
```

**Good:**
```python
# We need to mask out future positions for causal attention
# Upper triangular matrix excludes the diagonal (hence k=1)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

(See: [Don't Write Comments (YouTube)](https://www.youtube.com/watch?v=Bf7vDBBOBUA))


### Testing

The point of tests in this codebase is to ensure that the code is working as expected, not to prevent production outages - there's no deployment here.
Therefore, don't worry about lots of larger integration/end-to-end tests. These often require too much overhead for what it's worth in our case, and
this codebase is interactively run so often that issues will likely be caught by the user at very little cost.
