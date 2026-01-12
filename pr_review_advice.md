# Lessons from Dan & Oli's PR Reviews

Based on ~900 inline comments and ~350 general PR comments, here are the key themes and lessons.

## 1. Fight Over-Engineering Relentlessly

**The biggest recurring theme.** They consistently push back on unnecessary abstractions.

- "I don't think we need this protocol? can't we just pass X directly?"
- "This method is really big and handles all of these edge cases that we don't care about"
- "I'd really like to push on fewer lines of code and fewer of our own abstractions"
- "The Command class is overengineering here... just using shlex for injection safety"

**Actionable advice:**
- Before creating an abstraction, ask: "Does this justify its existence?"
- If a function only does meaningful work in one of its cases, inline it at call sites
- If handling an edge case adds significant complexity, don't handle it
- A researcher should understand the code quickly without learning custom abstractions

## 2. Comments: Less Is More

**Direct quote:** "Your first instinct should be: 'If I couldn't write any comments, how would I write this code?'"

Comments flagged for removal:
- "overkill comment imo" (appears 10+ times)
- "remove comment"
- Comments describing what code does rather than why

**Keep comments for:**
- Complex/non-obvious logic
- Explaining "why" not "what"
- Tensor shape documentation when shapes are complex

## 3. Types: Strict but Not Redundant

**Do:**
- Use proper types, never `list[Any]` - "module_info shouldn't be list[Any]"
- Use `T | None` instead of `Optional[T]`
- Use `dict`, `list`, `tuple` not `Dict`, `List`, `Tuple`
- Use kwargs for safety: "could you use kwargs instead of args here for safety?"

**Don't:**
- Add type annotations when obvious: `name: str = "John"` → just `name = "John"`
- Add Path type hints after `/` operator usage

## 4. Fail Fast, Not Silently

- "If there's an assumption you're making while writing code, assert it"
- "This feels like an unnecessary soft-fail" (on `ON CONFLICT DO NOTHING`)
- Add assertions for preconditions
- "I'd remove this line. We don't want this to succeed if elements is not in the computational graph"

**Pattern:**
```python
# Bad: silent handling
if ciThresholdInput === "": return;
const value = parseFloat(ciThresholdInput);
if (!isNaN(value)) ...

# Good: fail loudly
if (ciThresholdInput === "") return;
const value = parseFloat(ciThresholdInput);
if (isNaN(value)) throw new Error();
```

## 5. PR Hygiene

**PR descriptions:**
- Keep them concise and information-dense
- Include "Closes #X" to auto-close issues
- Link to evaluation runs/reports when relevant
- Use better PR names ("Refactor to use hooks" not generic names)

**Before merging:**
- "I'd like to see an evals run on all of tms and resid_mlp with a PR as big as this"
- "It'd be good to test whether this slows down training"
- "Link to the latest evals in the description"

## 6. Naming Consistency

- Variable names should match class names: "it's weird to me that the names are `run_clustering_config` when the class is `ClusteringRunConfig`"
- Avoid confusing short names: "otherwise the `c` can be confused with the `linear1_c`"
- Use `_id` suffix for foreign keys, `id` for primary keys
- Prefer `base_config_dict` or `base_config: dict` over just `base_config` when it's not a Config object

## 7. DRY (Don't Repeat Yourself)

- "This is defined twice in different tests. Maybe pull it out."
- "I'd probably prefer a decorator if using this in lots of functions"
- But don't create abstractions just for 2-3 similar lines - wait until the pattern is clear

## 8. Use NamedTuple/Dataclass for Complex Structures

When passing around tuples like `(source, target, c_in, c_out, s_in, s_out, strength, is_cross_seq)`:
- "can we NamedTuple or dataclass this?"

## 9. Test Thoughtfully

- "I'd use some different C values here, as I'm not sure the handling of that is tested elsewhere"
- "Put 999 or something, since this is never going to be used anyway"
- Test with minimal values where possible to speed up tests
- Don't over-test research code with integration tests

## 10. API Error Codes Matter

- "400 is weird for invalid json. Should just error out" vs "if the user provides a path to a dodgy file they get a clearer error"
- 400 = user error (bad input)
- 500 = internal error
- Choose based on who you consider the "source" of the error

## 11. Performance Awareness

- "This is the part that scares me... We're doing an all_reduce for each layer"
- "I think we should do a multi-node run... before and after the PR"
- "should we filter for CI > 0.0 here? I'm imagining this could be quite slow"

## 12. Config Migration: Keep It Simple

Bad pattern:
```python
def _migrate_to_module_info(cls, config_dict):
    # 60 lines handling every edge case
    ...
```

Good pattern:
```python
if "C" in config:
    logger.warning("Found 'C', mapping old structure to new")
    config_dict["module_info"] = [...]
    del config_dict["C"]
```

## Summary Checklist for PRs

Before submitting:
- [ ] Is every abstraction justified? Could I inline it?
- [ ] Are comments explaining "why" not "what"?
- [ ] Are types explicit where not obvious?
- [ ] Will the code fail loudly if assumptions are violated?
- [ ] Does the PR description include issue links and eval results?
- [ ] Are variable names clear and consistent?
- [ ] Is there duplicated code that should be extracted?
- [ ] Has performance impact been considered/tested?
- [ ] Are error codes appropriate (400 vs 500)?
- [ ] Is the migration/backwards-compat code minimal?
