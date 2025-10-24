# Dashboard Refactor TODO

This document tracks potential cleanup tasks discovered during the ZANJ refactor.

## Completed ✅

- Converted `DashboardData` to `SerializableDataclass`
- Added `ClusterSample` dataclass for self-contained samples
- Updated `ClusterData.generate()` to create self-contained samples
- Refactored JavaScript to use `ZanjLoader` instead of manual file loading
- Fixed `this.baseUrl` bug in `zanj.js`
- Simplified config.js to remove deprecated file paths

## Potential Cleanup Tasks

### Python Backend

1. **Remove deprecated fields in `ClusterData`** (`spd/clustering/dashboard/core/cluster_data.py:85`)
   - `criterion_samples` can be removed once JS is fully migrated
   - Currently kept for backward compatibility

2. **Remove deprecated methods in `DashboardData`** (`spd/clustering/dashboard/core/dashboard_data.py:102-103`)
   - Comment indicates deprecated methods can be removed
   - ZANJ handles serialization automatically via `SerializableDataclass`

3. **Simplify `ClusterData.serialize()`** (`spd/clustering/dashboard/core/cluster_data.py:259-298`)
   - Currently has manual serialization logic
   - Could potentially convert to `SerializableDataclass` if `.serialize()` isn't needed elsewhere
   - Check if manual `BinnedData` serialization is still required

4. **Remove `get_unique_activation_hashes()` method** (`spd/clustering/dashboard/core/cluster_data.py:217-225`)
   - No longer needed since activations are self-contained in samples
   - Was only used by old `DashboardData.save()` logic

### JavaScript Frontend

5. **Remove deprecated component activation logic** (`spd/clustering/dashboard/js/cluster-detail.js:11-12`)
   - `componentActivations` and `enabledComponents` variables
   - Related component toggle UI code (if no longer used)

6. **Remove `combineComponentActivations()` function** (if exists in cluster-detail.js)
   - Was used for combining component activations
   - No longer needed with self-contained samples

7. **Clean up config.js file paths** (`spd/clustering/dashboard/js/util/config.js:59-63`)
   - Remove commented-out deprecated file paths
   - Only `explanations` is needed now

8. **Alpine.js model info component simplification** (`spd/clustering/dashboard/js/cluster-selection.js:6-10`)
   - Could potentially be simplified further
   - Data is now set directly from ZANJ load

### Testing & Validation

9. **Test with existing dashboard data**
   - Verify backward compatibility with old data format (if needed)
   - Test explanations.jsonl loading still works

10. **Verify float16 handling**
    - Check if ZANJ preserves float16 dtype for activations
    - May need `serializable_field` configuration

11. **Test lazy loading behavior**
    - Verify large activation arrays lazy load correctly
    - Check memory usage with large datasets

### Documentation

12. **Update dashboard README** (if exists)
    - Document new ZANJ-based data format
    - Explain self-contained cluster structure
    - Update data generation examples

13. **Add ZANJ dependency to requirements**
    - Ensure `zanj` is in `pyproject.toml`
    - Document minimum version if applicable

## Notes

- The refactor maintains backward compatibility where possible via deprecated fields
- Component-level activation display may need revisiting if that feature is still used
- Consider adding progress indicators to ZANJ loading in the future
