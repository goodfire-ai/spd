# Code Style Guide

TLDR:
- prioritise simple, straightforward code. Our users are researchers, often with little coding experience.
- safety: use types, einops, jaxtyping, and liberal assertions.
- fail fast - if something is wrong, the code should fail, not recover silently.


## Design / Architecture

We want to decouple metrics and analysis from the core codebase as much as possible. Metric- and figure-generation code should be encapsulated in `spd/metrics.py` and `spd/figures.py`.

### Pure Functions & I/O
- Keep I/O at the edges of your code
- Make as many functions as possible pure (no side effects, deterministic output from inputs)
- Push file operations, logging, and network calls to the highest reasonable level in the call stack

### Data Structures
- **Don't use bare dictionaries for heterogeneous structures**
  - **Good**: `{user_id: user_data}` - homogeneous values indexed by keys
  - **Bad**: `{"tokens": [...], "loss": 0.5, "metadata": {...}}` - use a dataclass or typed dict instead
- This makes code more maintainable and catches errors at type-check time rather than runtime

### Fail Fast (Negative Space Programming)
Code should fail immediately when assumptions are violated, preventing bugs from propagating.

**Assert liberally:**
- If you have an invariant in your head, assert it
- If you're afraid to assert, your program might already be broken
- Never soft-fail or recover silently from unexpected states

**Don't handle errors gracefully without good reason:**
- If your program isn't working correctly, it shouldn't be running—you should be fixing it
- Exceptions should be exceptional; let the program crash and surface the real issue

## Type Annotations
- Use jaxtyping for tensor shapes (though for now we don't do runtime checking)
- Always use the PEP 604 typing format of `|` for unions and `type | None` over `Optional`
- Use `dict`, `list` and `tuple` not `Dict`, `List` and `Tuple`
- Don't add type annotations when they're redundant (i.e. `my_thing: Thing = Thing()` or `name: str = "John Doe"`)
- Prefer `match` over `if/elif/else` chains when dispatching on conditions - more declarative and makes cases explicit

### Encode Invariants in Types
Write your invariants into types as much as possible to offload validation to the type checker:

- **Joint dependencies**: If you need both `a` and `b` together, or neither:
  ```python
  # Bad: independently optional
  a: str | None
  b: int | None

  # Good: coupled in the type system
  config: tuple[str, int] | None
  ```

- **Make invalid states unrepresentable**: If tracking dependent states requires lots of assertions, encode the dependency in types instead

### Distinguish None from Empty Collections
Differentiate "no data" from "empty data" when semantically meaningful:

```python
# Bad: ambiguous
def process(items: list[str] | None = None): ...

# Good: explicit semantics
def process(items: list[str] | None):
    # None = data not loaded yet
    # [] = loaded but empty
    ...
```

## Tensor Operations
- Try to use einops by default for clarity
- Assert shapes liberally
- Document complex tensor manipulations

## Function Design

### Default Arguments
Default arguments are a code smell more often than they're useful:

- You should have a **very good reason** for having a default value for an argument
- Especially problematic when a caller also defaults to the same thing—this creates hidden coupling
- Keep defaults high in the call stack rather than deep in implementation details
- Often it's better to be explicit at the call site than to hide behavior in defaults

**Ask yourself**: "Does this default encode domain logic that should be visible to the caller?"

## Code Hygiene

### Delete Unused Code
- If an argument is always the same value, strongly consider removing it as a parameter
- Delete dead code immediately—don't comment it out
- If code isn't being used, the repository history will preserve it

### Remove Unnecessary Logs
- Logs should provide actionable information
- Remove debug logs that don't help users understand what's happening
- Keep logs at module boundaries and for genuine diagnostics

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
