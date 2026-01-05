# Fix: Prevent race condition in WandB file downloads

## Background

When using `simple_stories_train` models in multi-process contexts (e.g., `spd`'s `harvest_parallel` which spawns multiple GPU workers), each process calls `RunInfo.from_path()` to load the model. This triggers `_download_wandb_files()` which currently uses `replace=True` for all downloads:

```python
config_file.download(root=str(cache_dir), replace=True)
model_config_file.download(root=str(cache_dir), replace=True)
latest_ckpt_file.download(root=str(cache_dir), replace=True)
```

The `replace=True` parameter forces a re-download even when files already exist on disk. When multiple processes call this concurrently, they overwrite each other's downloads mid-stream, corrupting the checkpoint file.

This manifests as confusing errors like:
- `RuntimeError: Cannot use weights_only=True with files saved in the legacy .tar format` (partially written ZIP file misidentified as legacy format)
- `RuntimeError: PytorchStreamReader failed reading file data.pkl: file read failed` (corrupted/incomplete file)

## Solution

Check if files exist before downloading. This ensures files are downloaded once by the first process, and subsequent processes use the cached version.

## File to Modify

`simple_stories_train/run_info.py`

## Change

In `_download_wandb_files()`, replace lines 89-95:

```python
# Current code (lines 89-95):
config_file.download(root=str(cache_dir), replace=True)
model_config_file.download(root=str(cache_dir), replace=True)
latest_ckpt_file.download(root=str(cache_dir), replace=True)
if ln_stds_file is not None:
    ln_stds_file.download(root=str(cache_dir), replace=True)
if tokenizer_file is not None:
    tokenizer_file.download(root=str(cache_dir), replace=True)
```

With:

```python
# New code - skip download if file already exists:
if not (cache_dir / config_file.name).exists():
    config_file.download(root=str(cache_dir))
if not (cache_dir / model_config_file.name).exists():
    model_config_file.download(root=str(cache_dir))
if not (cache_dir / latest_ckpt_file.name).exists():
    latest_ckpt_file.download(root=str(cache_dir))
if ln_stds_file is not None and not (cache_dir / ln_stds_file.name).exists():
    ln_stds_file.download(root=str(cache_dir))
if tokenizer_file is not None and not (cache_dir / tokenizer_file.name).exists():
    tokenizer_file.download(root=str(cache_dir))
```

## Notes

- If you need to force a fresh download (e.g., model was updated on WandB), delete the cache directory at `{REPO_ROOT}/.cache/wandb_runs/{slug}/`
- An alternative would be to add a `force_download: bool = False` parameter to `RunInfo.from_path()`, but the simple existence check should suffice for most use cases
