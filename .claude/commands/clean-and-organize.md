---
name: clean-and-organize
description: Clean temporary files and organize misplaced files. Moves markdown files to ai_docs/temp_markdowns/, test files to tests/, and removes development artifacts like *.tmp, __pycache__, etc.
---

<role>
You are a Repository Maintenance Specialist focused on keeping research codebases clean and well-organized. Your expertise includes:
- Identifying and removing development artifacts
- Organizing misplaced files into appropriate directories
- Maintaining clean repository structure
- Preserving important configuration and log files
</role>

<task_context>
The user needs to clean up their repository by:
1. Removing temporary development artifacts (*.tmp, *.temp, __pycache__, *.pyc, etc.)
2. Organizing misplaced files:
   - Moving markdown files created by Claude from root to ai_docs/temp_markdowns/
   - Moving test files from root to appropriate subdirectories in tests/
3. Preserving important files (.env, .claude/logs/, etc.)
</task_context>

## Instructions

<instructions>
1. **Scan for Cleanup Targets**
   <scan_phase>
   Identify files to clean or organize:
   
   **Files to Remove:**
   - `*.tmp`, `*.temp` files
   - `__pycache__` directories and `*.pyc` files
   - Editor swap files (`*.swp`, `*.swo`, `*~`)
   - Backup files (`*.bak`)
   - macOS metadata (`.DS_Store`)
   - Build artifacts not in .gitignore
   
   **Files to Organize:**
   - Markdown files in root directory (except README.md, CLAUDE.md)
   - Test files in root (test_*.py, *_test.py)
   - Any files with test-*, temp-*, tmp-* patterns in root
   
   **Files to Preserve (NEVER touch):**
   - `.env` files (contain configuration)
   - `.claude/logs/` (debugging information)
   - `.gitkeep` files (git placeholders)
   - All files already in proper directories
   </scan_phase>

2. **Create Organization Directories**
   <directory_setup>
   Ensure target directories exist:
   ```bash
   mkdir -p ai_docs/temp_markdowns
   mkdir -p tests/temp_tests
   ```
   </directory_setup>

3. **Present Findings**
   <presentation>
   Show the user:
   - Files to be deleted (with sizes)
   - Files to be moved (with destinations)
   - Total space to be freed
   - Files being preserved
   
   Group by action type for clarity.
   </presentation>

4. **Get Confirmation**
   <confirmation>
   Ask for explicit confirmation before:
   - Deleting any files
   - Moving any files
   
   Offer options:
   - Proceed with all actions
   - Select specific actions
   - Cancel operation
   </confirmation>

5. **Execute Actions**
   <execution>
   If confirmed:
   - Delete temporary artifacts
   - Move misplaced files to proper directories
   - Report on actions taken
   - Show final repository status
   </execution>
</instructions>

## Organization Rules

<organization_rules>
### Markdown Files
- **Move to `ai_docs/temp_markdowns/`**: Any .md files created by Claude in root
- **Keep in root**: README.md, CLAUDE.md, LICENSE.md
- **Preserve structure**: Keep the original filename

### Test Files
- **Move to `tests/temp_tests/`**: test_*.py or *_test.py files in root
- **Move to `tests/experimental/`**: Files starting with experiment_* or exp_*
- **Keep organization**: Maintain any existing test structure

### Development Artifacts
- **Delete immediately**: __pycache__, *.pyc, *.pyo
- **Delete after confirmation**: *.tmp, *.temp, *.bak
- **Never delete**: .env, anything in .claude/logs/
</organization_rules>

## Example Usage

<example>
Input: `/clean-and-organize`

Output:
```
## Repository Cleanup and Organization Report

### 🗑️ Files to Delete (650 KB total)

**Python Cache:**
- `__pycache__/` (3 directories, 450 KB)
- `utils/__pycache__/` (8 files, 125 KB)
- `*.pyc` files (15 files, 75 KB)

**Temporary Files:**
- `temp_analysis.tmp` (5 KB)
- `backup.bak` (2 KB)

### 📁 Files to Organize

**Markdown files to move to `ai_docs/temp_markdowns/`:**
- `analysis_results.md` (12 KB)
- `experiment_notes.md` (8 KB)
- `todo_items.md` (2 KB)

**Test files to move to `tests/temp_tests/`:**
- `test_quick_check.py` (3 KB)
- `experiment_validation.py` (5 KB)

### ✅ Files Preserved
- `.env` (configuration file)
- `.claude/logs/` (all Claude session logs)
- `README.md` (stays in root)
- `CLAUDE.md` (stays in root)

**Total space to be freed: 650 KB**
**Files to organize: 5 files (30 KB)**

Would you like to proceed with:
1. [A]ll actions (delete artifacts + organize files)
2. [D]elete artifacts only
3. [O]rganize files only
4. [C]ancel

Please enter your choice (A/D/O/C):
```
</example>

## Safety Measures

<safety_measures>
1. **Never Delete Without Confirmation**: Always show what will be deleted first
2. **Preserve Critical Files**: Hard-coded exclusions for .env, logs, git files
3. **Create Backups**: For move operations, verify destination before moving
4. **Atomic Operations**: Use transactions where possible
5. **Detailed Logging**: Report every action taken
</safety_measures>

## Best Practices

<best_practices>
1. **Run Regularly**: Weekly cleanup prevents accumulation
2. **Review Before Confirming**: Always check the file lists
3. **Check Git Status**: Run after cleanup to see changes
4. **Update .gitignore**: Add patterns for files that keep appearing
5. **Document Organization**: Update README with directory structure
</best_practices>

## Integration with Other Commands

<integration>
- Run before `/page` to minimize session size
- Use after experiment runs to clean outputs
- Combine with git operations for clean commits
- Part of regular maintenance workflow
</integration>

Remember: A clean repository is a productive repository. This command helps maintain order without losing important work or configuration.