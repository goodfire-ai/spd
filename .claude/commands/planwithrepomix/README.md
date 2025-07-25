# Repomix + Gemini Integration for Claude Code

This directory contains slash commands and tools that integrate repomix with planwithgemini, enabling Claude Code to efficiently analyze entire codebases and create comprehensive implementation plans.

## Overview

The integration combines:
- **Repomix**: Efficiently packs code repositories into AI-friendly formats
- **Gemini**: Analyzes large contexts to generate detailed plans
- **Claude Code**: Orchestrates the workflow through custom slash commands

## Commands

### `/plan-auto-context`

Intelligently selects relevant files based on your task description.

**Usage:**
```
/plan-auto-context implement caching layer for API endpoints
```

**Features:**
- Smart keyword extraction from task description
- Automatic file filtering based on relevance
- Optimized token usage (typically 70-80% reduction)
- Best for focused features and specific implementations

### `/plan-full`

Analyzes the entire codebase for comprehensive understanding.

**Usage:**
```
/plan-full refactor authentication system to use OAuth2
```

**Features:**
- Full codebase analysis with compression
- System-wide impact assessment
- Architecture-level insights
- Best for major refactors and architectural changes

## Quick Start

### From Claude Code

1. **For focused planning:**
   ```
   /plan-auto-context add user profile feature
   ```

2. **For major changes:**
   ```
   /plan-full migrate from REST to GraphQL
   ```

### From Command Line

Using the shell script:
```bash
# Auto mode (smart selection)
./scripts/repomix/planwithgemini.sh auto "implement Redis caching"

# Full analysis
./scripts/repomix/planwithgemini.sh full "refactor to microservices" -o refactor-plan.md

# Security audit
./scripts/repomix/planwithgemini.sh security "audit authentication"
```

Using the Python script:
```bash
# Auto mode with verbose output
python scripts/repomix/planwithgemini.py auto "add payment processing" -v

# Full analysis with custom config
python scripts/repomix/planwithgemini.py full "database migration" \
  -c custom-config.json -o db-migration-plan.md

# Custom mode with specific includes
python scripts/repomix/planwithgemini.py custom "analyze performance" \
  -- --include "**/perf/**" --include "**/metrics/**"
```

## Configuration

The `repomix-gemini.config.json` file contains optimized settings for different scenarios:

### Key Settings

```json
{
  "output": {
    "format": "xml",           // Best format for AI comprehension
    "xmlStyle": "structured",  // Organized structure
    "headerComments": true,    // Include file metadata
    "topFilesCount": 50       // Show most relevant files
  },
  "compression": {
    "enabled": true,          // Reduce token usage
    "level": 6,               // Balance speed/compression
    "threshold": 50000        // Compress large files
  },
  "gemini": {
    "maxTokens": 2000000,     // Gemini's context limit
    "compressionTargets": {   // Size-based strategies
      "small": 100000,
      "medium": 500000,
      "large": 1000000,
      "xlarge": 2000000
    }
  }
}
```

### Presets

- **quick**: Maximum compression for rapid analysis
- **detailed**: Full detail with line numbers
- **security**: Focus on security-relevant files

## Modes Explained

### Auto Mode
- Extracts keywords from your task description
- Includes files matching those keywords
- Excludes test files by default
- Ideal for: New features, bug fixes, small refactors

### Full Mode
- Packs entire codebase with smart exclusions
- Uses maximum compression for large codebases
- Provides complete architectural view
- Ideal for: Major refactors, migrations, audits

### Quick Mode
- Aggressive compression (removes comments, empty lines)
- Fastest analysis time
- Good overview without details
- Ideal for: Quick assessments, feasibility checks

### Security Mode
- Focuses on auth, security, and config files
- Includes security checks
- Highlights potential vulnerabilities
- Ideal for: Security audits, compliance checks

### Custom Mode
- Full control over repomix parameters
- Use any repomix CLI options
- Combine with Gemini analysis
- Ideal for: Specific requirements, unique workflows

## Examples

### Example 1: Adding a Feature
```bash
# Claude Code
/plan-auto-context implement user notifications with email and SMS

# Repomix will automatically include:
# - Files containing "notification", "email", "sms"
# - Related service and controller files
# - Configuration files
```

### Example 2: Major Refactoring
```bash
# Claude Code
/plan-full refactor monolithic application to microservices architecture

# Provides:
# - Complete dependency analysis
# - Service boundary recommendations
# - Migration phases with timelines
# - Risk assessment and rollback plans
```

### Example 3: Performance Analysis
```bash
# Command line with custom patterns
python scripts/repomix/planwithgemini.py custom "optimize database queries" \
  -- --include "**/models/**" --include "**/queries/**" --include "**/*repository*"
```

## Best Practices

1. **Start with Auto Mode**: Usually provides enough context for most tasks
2. **Use Full Mode Sparingly**: Only for architectural decisions
3. **Save Important Plans**: Use `-o` flag to save plans as markdown
4. **Iterate on Keywords**: If auto mode misses files, add keywords to your description
5. **Check Token Count**: Repomix shows token usage - aim to stay under 1M for best results

## Troubleshooting

### "repomix not found"
```bash
npm install -g repomix
```

### "gemini not found"
Install the Gemini CLI from the official repository.

### Context too large
- Use higher compression levels
- Exclude more file types
- Use auto mode instead of full
- Split analysis into sections

### Missing relevant files
- Add more specific keywords to your task
- Use custom mode with explicit includes
- Check if files match exclude patterns

## Advanced Usage

### Combining with Other Commands

1. **With /setup**:
   ```
   /setup
   /plan-full integrate new codebase architecture
   ```

2. **With /parallel-analysis-example**:
   ```
   /plan-auto-context design API endpoints
   /parallel-analysis-example analyze API design from security, performance, and usability perspectives
   ```

3. **With /page**:
   ```
   /plan-full major refactoring
   /page refactoring-plan-checkpoint
   ```

### Remote Repository Analysis
```bash
# Analyze without cloning
python scripts/repomix/planwithgemini.py auto "understand architecture" \
  -- --remote https://github.com/org/repo
```

### Progressive Analysis
```bash
# Start with structure
repomix --files-only . > structure.txt

# Then analyze critical paths
/plan-auto-context implement based on structure
```

## Performance Tips

1. **For Large Codebases (>100K files)**:
   - Use compression level 9
   - Enable comment removal for non-documentation tasks
   - Consider splitting into subsystems

2. **For Quick Iterations**:
   - Keep context files with `--keep-context`
   - Reuse for multiple analyses
   - Compare before/after states

3. **For CI/CD Integration**:
   - Use Python script for better error handling
   - Output to files for artifact storage
   - Parse token counts for monitoring

## Integration Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────┐
│ Claude Code │────▶│ Repomix  │────▶│ Gemini  │
│   (slash    │     │  (file   │     │ (large  │
│  commands)  │     │ packing) │     │context) │
└─────────────┘     └──────────┘     └─────────┘
       │                  │                 │
       └──────────────────┴─────────────────┘
                          │
                   ┌──────────────┐
                   │ Plan Output  │
                   │  (markdown)  │
                   └──────────────┘
```

## Contributing

To add new modes or improve the integration:

1. Edit the command files in `.claude/commands/planwithrepomix/`
2. Update the scripts in `scripts/repomix/`
3. Test with various codebase sizes
4. Update this README with new features

## Future Enhancements

- [ ] MCP server integration for streaming updates
- [ ] Caching layer for repeated analyses
- [ ] Differential analysis (what changed since last plan)
- [ ] Multi-language prompt templates
- [ ] Integration with other AI models
- [ ] Plan execution tracking

---

Remember: The goal is to get the right context to Gemini efficiently. Start small, expand as needed, and let the tools handle the complexity.