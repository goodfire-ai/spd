# Autonomous Research Scripts

This directory contains programmatic versions of Claude slash commands that can be executed autonomously for AI safety research workflows.

## Overview

These scripts enable:
- **Autonomous execution** of complex research tasks
- **Multi-model orchestration** (Claude, Gemini, O3, etc.)
- **Codebase integration** from GitHub or local sources
- **Parallel analysis** with specialized agents
- **Specification-driven development** with automatic test generation

## Available Scripts

### 🔧 Core Commands

#### 1. `integrate_codebase.py`
Integrate external codebases for analysis.

```bash
# From GitHub
python scripts/integrate_codebase.py https://github.com/org/repo.git

# From local path
python scripts/integrate_codebase.py /path/to/codebase --name my-project
```

#### 2. `commands/multi_mind.py`
Run parallel analysis with multiple specialized agents.

```bash
# Default 6 agents
python scripts/commands/multi_mind.py "Analyze security vulnerabilities in authentication system"

# Custom agents
python scripts/commands/multi_mind.py "Task description" --agents custom_agents.json --output ./results
```

#### 3. `commands/spec_driven.py`
Generate specifications and test suites.

```bash
# Basic usage
python scripts/commands/spec_driven.py "User authentication system"

# With requirements file
python scripts/commands/spec_driven.py "Payment processing" --requirements requirements.txt --output ./specs
```

#### 4. `commands/analyze_function.py`
Deep function analysis for complexity and security.

```bash
# Full analysis
python scripts/commands/analyze_function.py process_payment --codebase ./project

# Metrics only
python scripts/commands/analyze_function.py main --codebase ./project --metrics-only
```

#### 5. `commands/orchestrate_research.py`
Coordinate multi-tool research workflows.

```bash
# Predefined workflow
python scripts/commands/orchestrate_research.py \
    --workflow comprehensive_analysis \
    --codebase ./project \
    --task "Analyze security architecture"

# Custom workflow
python scripts/commands/orchestrate_research.py \
    --custom-steps workflow.json \
    --codebase ./project
```

## Workflow Examples

### Security Analysis Workflow
```json
{
  "name": "security_audit",
  "steps": [
    {
      "tool": "multi_mind",
      "args": {
        "agents": [
          {"role": "Security Analyst", "focus": "vulnerabilities"},
          {"role": "Cryptographer", "focus": "encryption"},
          {"role": "Network Security", "focus": "communications"}
        ]
      }
    },
    {
      "tool": "gemini",
      "args": {"query": "Analyze attack vectors"}
    },
    {
      "tool": "spec_driven",
      "args": {"feature": "security patches"}
    }
  ]
}
```

### Architecture Review Workflow
```bash
# 1. Integrate codebase
python scripts/integrate_codebase.py https://github.com/project/repo.git

# 2. Run comprehensive analysis
python scripts/commands/orchestrate_research.py \
    --workflow comprehensive_analysis \
    --codebase external_codebases/repo \
    --task "Architecture and design pattern analysis" \
    --report architecture_report.md

# 3. Generate improvement specs
python scripts/commands/spec_driven.py \
    "Refactored architecture based on analysis" \
    --requirements architecture_report.md
```

## Custom Agent Configuration

Create `agents.json`:
```json
[
  {
    "role": "AI Safety Researcher",
    "focus": "Analyze alignment mechanisms and failure modes"
  },
  {
    "role": "Security Auditor", 
    "focus": "Identify potential misuse vectors"
  },
  {
    "role": "Performance Engineer",
    "focus": "Assess computational efficiency and scaling"
  }
]
```

Use with:
```bash
python scripts/commands/multi_mind.py "Analyze model" --agents agents.json
```

## Environment Setup

### Required Tools
- Python 3.8+
- Claude CLI (`pip install claude`)
- Gemini CLI (optional): `pip install gemini-cli`
- Git (for repository integration)

### Configuration
Create `.claude/orchestration_config.json`:
```json
{
  "tools": {
    "claude": {"command": "claude", "max_tokens": 100000},
    "gemini": {"command": "gemini", "model": "gemini-pro"}
  },
  "workflows": {
    "custom_workflow": [...]
  }
}
```

## Best Practices

1. **Start with Integration**: Always integrate codebases first for better context
2. **Use Workflows**: Combine tools for comprehensive analysis
3. **Save Results**: Use `--output` flags to preserve findings
4. **Iterate**: Use results from one tool as input to another
5. **Verify**: Cross-reference findings across multiple tools

## Error Handling

Common issues and solutions:

### Command Not Found
```bash
export PATH=$PATH:$(pwd)/scripts/commands
```

### Rate Limiting
Add delays in orchestration config:
```json
{
  "step_delay": 5,
  "max_concurrent": 3
}
```

### Large Codebases
Use focused analysis:
```bash
# Analyze specific module
python scripts/commands/analyze_function.py function_name \
    --codebase ./project/module
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Security Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        
      - name: Install Dependencies
        run: |
          pip install claude-cli
          pip install -r requirements.txt
          
      - name: Run Security Analysis
        run: |
          python scripts/commands/orchestrate_research.py \
            --workflow security_audit \
            --codebase . \
            --report security_report.md
            
      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: security-report
          path: security_report.md
```

## Contributing

To add new autonomous commands:

1. Create script in `scripts/commands/`
2. Follow the existing pattern with argparse
3. Support both interactive and programmatic use
4. Add to orchestration workflows
5. Document in this README

## License

These scripts are part of the AI Safety Research toolkit and are provided under the same license as the parent project.