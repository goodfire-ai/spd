# Repomix + Gemini Integration Examples

This document provides practical examples of using the repomix-gemini integration for various planning scenarios.

## Basic Usage Examples

### 1. Adding a New Feature

**Scenario**: You need to add a notification system to your application.

**Claude Code Command**:
```
/plan-auto-context implement notification system with email, SMS, and in-app notifications
```

**What happens**:
- Repomix identifies keywords: "notification", "email", "sms", "app"
- Automatically includes files matching these patterns
- Gemini analyzes the context and creates an implementation plan
- You get a structured plan with phases, file modifications, and risk assessment

**Command Line Alternative**:
```bash
python scripts/repomix/planwithgemini.py auto \
  "implement notification system with email, SMS, and in-app notifications"
```

### 2. Major Refactoring

**Scenario**: Converting a monolithic application to microservices.

**Claude Code Command**:
```
/plan-full refactor monolithic application to microservices architecture
```

**What happens**:
- Repomix packs the entire codebase with compression
- Gemini performs comprehensive analysis
- Delivers detailed migration strategy with phases
- Identifies service boundaries and dependencies

**Command Line Alternative**:
```bash
./scripts/repomix/planwithgemini.sh full \
  "refactor monolithic application to microservices" \
  -o microservices-migration-plan.md
```

### 3. Security Audit

**Scenario**: Auditing API endpoints for security vulnerabilities.

**Claude Code Command**:
```
/plan-auto-context security audit all API endpoints for authentication and authorization
```

**Command Line Alternative**:
```bash
python scripts/repomix/planwithgemini.py security \
  "audit API endpoints for authentication vulnerabilities"
```

## Advanced Examples

### 4. Performance Optimization

**Using custom mode to focus on specific areas**:

```bash
# Focus on database and caching layers
python scripts/repomix/planwithgemini.py custom \
  "optimize database queries and caching strategy" \
  -- --include "**/models/**" \
     --include "**/cache/**" \
     --include "**/queries/**" \
     --include "**/*repository*"
```

### 5. Specific File Refactoring

**When you know exactly which files need refactoring**:

```bash
# Direct file refactoring
python scripts/repomix/claude-refactor.py \
  "extract authentication logic into separate service" \
  src/api/routes.py \
  src/middleware/auth.py \
  src/utils/tokens.py
```

### 6. Architecture Analysis

**Understanding an existing codebase**:

```bash
# Quick architecture overview
./scripts/repomix/quick-plan.sh "analyze and document current architecture"

# Detailed with specific focus
python scripts/repomix/planwithgemini.py full \
  "analyze architecture focusing on scalability bottlenecks" \
  --keep-context \
  -o architecture-analysis.md
```

## Integration Patterns

### 7. Progressive Analysis

**Start broad, then drill down**:

```bash
# Step 1: Get overview
/plan-quick understand overall system architecture

# Step 2: Focus on problem area
/plan-auto-context optimize user authentication flow

# Step 3: Detailed refactoring plan
/plan-full modernize authentication to use JWT tokens
```

### 8. Combining with Other Claude Commands

**Multi-command workflow**:

```
# 1. Set up the project
/setup

# 2. Integrate external authentication library
/integrate-external-codebase https://github.com/auth/library

# 3. Plan the integration
/plan-auto-context integrate external auth library with existing user system

# 4. Create parallel analysis
/parallel-analysis-example analyze authentication approach from security, performance, and user experience perspectives
```

### 9. Remote Repository Analysis

**Analyze without cloning**:

```bash
# Analyze a GitHub repository directly
python scripts/repomix/planwithgemini.py auto \
  "understand authentication implementation" \
  -- --remote https://github.com/example/repo
```

### 10. Iterative Planning

**Refine your plan based on initial results**:

```bash
# Initial broad analysis
./scripts/repomix/planwithgemini.sh auto \
  "implement real-time features" \
  --keep-context

# Review the plan, then get more specific
python scripts/repomix/planwithgemini.py custom \
  "implement WebSocket-based real-time chat" \
  -- --include "**/websocket/**" \
     --include "**/chat/**" \
     --include "**/realtime/**"
```

## Common Patterns by Use Case

### For New Features
```bash
# Smart selection based on feature description
/plan-auto-context implement [feature description with keywords]
```

### For Bug Fixes
```bash
# Focus on specific area
python scripts/repomix/planwithgemini.py auto \
  "fix memory leak in image processing pipeline" \
  -i "**/image/**" -i "**/memory/**"
```

### For Code Reviews
```bash
# Comprehensive analysis
/plan-full review codebase for best practices and potential improvements
```

### For Documentation
```bash
# Generate documentation plan
./scripts/repomix/planwithgemini.sh auto \
  "create comprehensive API documentation"
```

### For Testing
```bash
# Test strategy planning
/plan-auto-context create comprehensive test suite for payment processing
```

## Tips and Tricks

### 1. Keyword Optimization
Include specific technical terms in your task description:
- ❌ "make it faster"
- ✅ "optimize database queries and implement Redis caching"

### 2. Save Important Plans
Always save critical plans:
```bash
python scripts/repomix/planwithgemini.py full \
  "critical refactoring task" \
  -o plans/refactor-$(date +%Y%m%d).md
```

### 3. Use Compression Wisely
- For large codebases (>50MB), use full mode with compression
- For focused tasks (<10MB context), auto mode without compression is faster

### 4. Combine Modes
```bash
# Get structure first
repomix --files-only . > structure.txt

# Then analyze specific parts
/plan-auto-context implement feature based on structure
```

### 5. Monitor Token Usage
Check the token count in output:
- <500K tokens: Use auto mode
- 500K-1M tokens: Good for most analyses
- >1M tokens: Consider splitting or using more compression

## Troubleshooting Examples

### Context Too Large
```bash
# Solution 1: Increase compression
python scripts/repomix/planwithgemini.py quick "your task"

# Solution 2: Focus on subdirectory
python scripts/repomix/planwithgemini.py auto "your task" -t src/specific-module

# Solution 3: Exclude more files
python scripts/repomix/planwithgemini.py auto "your task" \
  -e "**/*.test.js" -e "**/*.spec.js" -e "**/docs/**"
```

### Missing Important Files
```bash
# Solution: Add explicit includes
python scripts/repomix/planwithgemini.py auto \
  "implement caching" \
  -i "**/important-file.js" \
  -i "**/critical-directory/**"
```

### Need More Detail
```bash
# Switch from quick to auto or full
/plan-full provide detailed implementation plan for complex feature
```

Remember: The key to effective planning is providing clear task descriptions with relevant keywords and choosing the appropriate mode for your needs.