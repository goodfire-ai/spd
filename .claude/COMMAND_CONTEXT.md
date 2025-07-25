# Command Context Loader

This file ensures Claude Code has access to all custom commands for autonomous execution.

## Core Template Commands

### Setup & Management
- @.claude/commands/setup.md - Interactive setup wizard for new projects
- @.claude/commands/crud-claude-commands.md - Create custom commands for your research
- @.claude/commands/page.md - Session state preservation
- @.claude/commands/plan-with-context.md - Smart implementation planning

### Example Patterns
- @.claude/commands/parallel-analysis-example.md - Multi-agent analysis pattern

## Research Documentation

### Core Guides
- @docs/guides/RESEARCH_PRINCIPLES.md - Universal research coding principles
- @docs/guides/RESEARCH_COMMAND_GUIDE.md - Command selection for research tasks
- @docs/examples/RESEARCH_EXAMPLES.md - Examples across research domains
- @docs/reference/RESEARCH_QUICK_REFERENCE.md - Quick reference for research workflows

## Script Implementations
- @scripts/commands/multi_mind.py - Autonomous multi-mind execution
- @scripts/commands/analyze_function.py - Programmatic function analysis
- @scripts/commands/spec_driven.py - Automated spec generation
- @scripts/commands/orchestrate_research.py - Multi-tool orchestration

## Integration Scripts
- @scripts/integrate_codebase.py - External codebase integration
- @scripts/README.md - Complete script documentation

## Loading Instructions

Claude Code should:
1. Reference these files when making decisions about command usage
2. Load specific command files when their triggers are detected
3. Use the programmatic scripts for autonomous execution
4. Follow the decision guide for automatic command selection

## Context Management

When context usage reaches:
- 50%: Start planning for eventual paging
- 70%: Actively suggest /page command
- 80%: Automatically execute /page with descriptive name
- 90%: Force page immediately to prevent loss