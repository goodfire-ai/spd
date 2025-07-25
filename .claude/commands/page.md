---
name: page
description: Save your work before context fills up. Example - /page experiment-checkpoint-1. Use when context >70% or switching tasks. Allows you to resume later with claude --resume
---

<role>
You are a Session Archivist specializing in preserving conversation context and enabling seamless continuity. Your expertise includes:
- Memory management and compression
- Context preservation strategies
- Information hierarchy and summarization
- Citation and reference tracking
- Session state serialization
</role>

<task_context>
The user needs to save their current session state before the context window fills up. This command implements OS-style paging to swap out conversation memory while preserving the ability to resume with full context later.
</task_context>

## Instructions

<instructions>
1. **Assess Current State**
   <state_assessment>
   - Calculate current context usage percentage
   - Identify key topics and outcomes from conversation
   - Determine critical information to preserve
   - Note any incomplete tasks or pending items
   </state_assessment>

2. **Extract Session Information**
   <extraction_process>
   Gather:
   - Complete conversation history with timestamps
   - All code snippets and their outcomes
   - Key decisions and rationale
   - External references and citations
   - Error messages and resolutions
   - Current working state and variables
   </extraction_process>

3. **Create Preservation Artifacts**
   <artifacts>
   Generate three levels of preservation:
   
   a) **Full History** (page_NAME_full.json)
      - Complete message history
      - All tool calls and responses
      - Timestamps and metadata
      - Searchable format
   
   b) **Compact Summary** (page_NAME_summary.md)
      - Executive summary of session
      - Key outcomes and decisions
      - Important code snippets
      - Next steps and todos
   
   c) **Resume Context** (page_NAME_resume.json)
      - Minimal context for continuation
      - Current state snapshot
      - Active variables and config
      - Pending operations
   </artifacts>

4. **Create Citations**
   <citation_system>
   For each important element:
   - Message reference: [Conv:Page:MessageID]
   - Code reference: [Code:Page:BlockID]
   - Decision reference: [Decision:Page:ID]
   - Enable back-references in future sessions
   </citation_system>

5. **Save and Index**
   <save_process>
   - Save to ~/.claude/pages/{{page_name}}/
   - Update search index for quick retrieval
   - Create symlink to latest page
   - Log in session activity tracker
   </save_process>
</instructions>

## Page Naming Convention

<naming_convention>
Use descriptive names following this pattern:
- Format: `YYYY-MM-DD-topic-checkpoint`
- Examples:
  - `2024-03-15-ml-training-v1`
  - `2024-03-15-api-refactor-complete`
  - `2024-03-15-debug-memory-leak`

Or let the system auto-generate:
- `page-{{timestamp}}-{{topic_hash}}`
</naming_convention>

## Output Format

<output_format>
```markdown
# Session Paged: {{page_name}}

## Summary
{{2-3 paragraph summary of session achievements}}

## Key Outcomes
1. {{Major achievement or decision}}
2. {{Important discovery or solution}}
3. {{Significant code implementation}}

## Artifacts Created
- Full History: `~/.claude/pages/{{page_name}}/full.json` ({{size}})
- Summary: `~/.claude/pages/{{page_name}}/summary.md` ({{size}})
- Resume Context: `~/.claude/pages/{{page_name}}/resume.json` ({{size}})

## Code Snippets Preserved
1. {{Description}}: [Code:{{page_name}}:{{id}}]
2. {{Description}}: [Code:{{page_name}}:{{id}}]

## Pending Tasks
- [ ] {{Incomplete task 1}}
- [ ] {{Incomplete task 2}}

## Resume Command
To continue this session later:
```bash
claude --resume {{page_name}}
```

Or reference specific elements:
```bash
# Get specific code snippet
claude "Show me [Code:{{page_name}}:{{id}}]"

# Review specific decision
claude "Explain [Decision:{{page_name}}:{{id}}]"
```

## Related Pages
- Previous: {{previous_page_name}}
- Related: {{related_topic_pages}}
```
</output_format>

## Integration with Session Management

<session_integration>
The paging system integrates with:

1. **extract-claude-session.py**:
   ```bash
   python scripts/extract-claude-session.py --page {{page_name}}
   ```

2. **Session Search**:
   ```bash
   claude-search --page {{page_name}} "search terms"
   ```

3. **Session Diff**:
   ```bash
   claude-diff --pages page1 page2
   ```

4. **Session Merge**:
   ```bash
   claude-merge --pages page1 page2 --output merged
   ```
</session_integration>

## Advanced Features

<advanced_features>
### Selective Paging
Page only specific parts of the conversation:
- `--code-only`: Save only code snippets
- `--decisions-only`: Save only key decisions
- `--after-message N`: Save from message N onward

### Compression Options
- `--compress`: Use gzip compression
- `--dedupe`: Remove duplicate content
- `--minimize`: Keep only essential context

### Export Formats
- `--format json`: Machine-readable format
- `--format markdown`: Human-readable summary
- `--format pdf`: Printable documentation
</advanced_features>

## Best Practices

<best_practices>
1. **Page Proactively**: Don't wait until 95% full
2. **Use Descriptive Names**: Makes resuming easier
3. **Document Checkpoints**: Note why you paged
4. **Chain Pages**: Reference previous pages in new ones
5. **Clean Up**: Archive old pages periodically

### When to Page
- Context usage > 80%
- Completing major milestone
- Switching to different task
- Before complex operations
- End of work session
</best_practices>

## Memory Management Strategy

<memory_strategy>
Think of paging like OS memory management:

1. **Working Set**: Current active context
2. **Page File**: Saved conversation states
3. **Page Fault**: Need to resume old context
4. **Cache**: Recent pages for quick access

Optimal strategy:
- Keep working set focused
- Page completed work immediately
- Maintain page index for quick lookup
- Use semantic page names
</memory_strategy>

## Example Usage

<example>
Input: `/page "ml-model-training-complete"`

Output:
```markdown
# Session Paged: ml-model-training-complete

## Summary
Successfully implemented and trained a sparse autoencoder for interpretability research. Achieved 94% reconstruction accuracy with 15% active features. Resolved gradient instability issues through adaptive learning rate scheduling and gradient clipping.

## Key Outcomes
1. Implemented custom sparse autoencoder with L1 regularization
2. Fixed NaN gradient issue with proper initialization
3. Achieved target sparsity while maintaining reconstruction quality
4. Created reusable training pipeline with checkpointing

## Artifacts Created
- Full History: `~/.claude/pages/ml-model-training-complete/full.json` (2.4MB)
- Summary: `~/.claude/pages/ml-model-training-complete/summary.md` (18KB)
- Resume Context: `~/.claude/pages/ml-model-training-complete/resume.json` (156KB)

## Code Snippets Preserved
1. Sparse Autoencoder Implementation: [Code:ml-model-training-complete:1]
2. Training Loop with Gradient Clipping: [Code:ml-model-training-complete:2]
3. Visualization Functions: [Code:ml-model-training-complete:3]

## Pending Tasks
- [ ] Run ablation studies on sparsity levels
- [ ] Implement feature importance analysis
- [ ] Create paper figures

## Resume Command
To continue this session later:
```bash
claude --resume ml-model-training-complete
```
```
</example>

Remember: Effective paging preserves not just the conversation, but the context and continuity needed to resume productive work. Think of each page as a checkpoint in a long journey of discovery.