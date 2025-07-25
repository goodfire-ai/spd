#!/bin/bash

# planwithgemini.sh - Intelligent planning with repomix and Gemini
# Usage: ./planwithgemini.sh [mode] "task description" [options]
# Modes: auto, full, quick, security, custom

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEMP_DIR=".temp"
CONFIG_FILE="repomix-gemini.config.json"
DEFAULT_MODE="auto"

# Create temp directory if it doesn't exist
mkdir -p "$TEMP_DIR"

# Function to display help
show_help() {
    echo "planwithgemini - Intelligent planning with repomix and Gemini"
    echo ""
    echo "Usage: $0 [mode] \"task description\" [options]"
    echo ""
    echo "Modes:"
    echo "  auto     - Smart file selection based on task (default)"
    echo "  full     - Full codebase analysis"
    echo "  quick    - Compressed analysis for quick planning"
    echo "  security - Security-focused analysis"
    echo "  custom   - Use custom repomix arguments"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE    Use custom config file"
    echo "  -o, --output FILE    Save plan to file"
    echo "  -i, --include PATTERN Additional include patterns"
    echo "  -e, --exclude PATTERN Additional exclude patterns"
    echo "  -v, --verbose        Verbose output"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 auto \"implement caching layer\""
    echo "  $0 full \"refactor authentication system\" -o auth-refactor-plan.md"
    echo "  $0 security \"audit API endpoints\""
    echo "  $0 custom \"analyze performance\" -- --include \"**/perf/**\""
}

# Function to extract keywords from task description
extract_keywords() {
    local task="$1"
    # Extract potential file/directory names and technical terms
    echo "$task" | tr '[:upper:]' '[:lower:]' | \
        grep -oE '[a-z]+' | \
        grep -vE '^(the|and|or|for|with|from|to|in|on|at|of|a|an)$' | \
        sort -u
}

# Function to run repomix with smart selection
run_repomix_auto() {
    local task="$1"
    local output_file="$2"
    local keywords=$(extract_keywords "$task")
    
    echo -e "${BLUE}Analyzing task for smart file selection...${NC}"
    echo "Keywords identified: $keywords"
    
    # Build include patterns based on keywords
    local includes=""
    for keyword in $keywords; do
        includes="$includes --include \"**/*${keyword}*\" --include \"**/${keyword}/**\""
    done
    
    # Run repomix with smart selection
    local cmd="repomix --format xml --output \"$output_file\" --stats --token-count"
    cmd="$cmd --config \"$CONFIG_FILE\""
    cmd="$cmd $includes"
    cmd="$cmd --compression-level 6"
    cmd="$cmd ."
    
    echo -e "${YELLOW}Running: $cmd${NC}"
    eval $cmd
}

# Function to run repomix for full analysis
run_repomix_full() {
    local output_file="$1"
    
    echo -e "${BLUE}Packing entire codebase for comprehensive analysis...${NC}"
    
    repomix --format xml \
            --output "$output_file" \
            --config "$CONFIG_FILE" \
            --compression-level 9 \
            --stats \
            --token-count \
            .
}

# Function to run repomix in quick mode
run_repomix_quick() {
    local output_file="$1"
    
    echo -e "${BLUE}Running quick compressed analysis...${NC}"
    
    repomix --format xml \
            --output "$output_file" \
            --config "$CONFIG_FILE" \
            --compression-level 9 \
            --remove-comments \
            --remove-empty-lines \
            --stats \
            .
}

# Function to run repomix for security analysis
run_repomix_security() {
    local output_file="$1"
    
    echo -e "${BLUE}Running security-focused analysis...${NC}"
    
    repomix --format xml \
            --output "$output_file" \
            --config "$CONFIG_FILE" \
            --include "**/auth/**" \
            --include "**/security/**" \
            --include "**/*config*" \
            --include "**/*.env*" \
            --security-check \
            --stats \
            .
}

# Function to create Gemini prompt based on mode and task
create_gemini_prompt() {
    local mode="$1"
    local task="$2"
    
    case $mode in
        auto|quick)
            echo "Analyze this codebase context and create a detailed implementation plan for: $task

Focus on:
1. Understanding current architecture and patterns
2. Identifying integration points for the new feature
3. Potential risks and dependencies
4. Step-by-step implementation approach with specific code changes
5. Testing and validation strategy
6. Performance considerations

Provide a structured plan with:
- Executive summary
- Implementation phases with time estimates
- Specific files to modify with rationale
- Risk assessment and mitigation
- Success criteria"
            ;;
        full)
            echo "Perform a comprehensive analysis of this entire codebase for: $task

Provide:
1. Current Architecture Assessment
   - Design patterns and principles in use
   - Technical debt and improvement areas
   - Component coupling and cohesion analysis

2. Implementation Strategy
   - Detailed refactoring/implementation plan
   - Migration approach if applicable
   - Backward compatibility considerations

3. System-wide Impact Analysis
   - Direct and indirect effects
   - Breaking changes identification
   - Performance implications

4. Risk Management
   - Critical risks and mitigation strategies
   - Rollback procedures
   - Incremental deployment approach

5. Testing and Validation
   - Comprehensive testing strategy
   - Performance benchmarks
   - Monitoring requirements

Structure as a detailed, actionable plan with phases, milestones, and specific tasks."
            ;;
        security)
            echo "Perform a security-focused analysis for: $task

Examine:
1. Authentication and Authorization
   - Current implementation review
   - Vulnerabilities and weaknesses
   - Best practice compliance

2. Data Security
   - Sensitive data handling
   - Encryption usage
   - Data flow analysis

3. API Security
   - Endpoint protection
   - Input validation
   - Rate limiting and abuse prevention

4. Security Improvements
   - Prioritized recommendations
   - Implementation approach
   - Testing requirements

Provide actionable security improvements with implementation details."
            ;;
        *)
            echo "Analyze this codebase and provide insights for: $task"
            ;;
    esac
}

# Function to run Gemini analysis
run_gemini_analysis() {
    local context_file="$1"
    local prompt="$2"
    local output_file="$3"
    
    echo -e "${BLUE}Running Gemini analysis...${NC}"
    
    if [ -n "$output_file" ]; then
        gemini -p "@$context_file" "$prompt" > "$output_file"
        echo -e "${GREEN}Plan saved to: $output_file${NC}"
    else
        gemini -p "@$context_file" "$prompt"
    fi
}

# Main execution
main() {
    local mode="$DEFAULT_MODE"
    local task=""
    local output_file=""
    local verbose=false
    local custom_args=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            auto|full|quick|security|custom)
                mode="$1"
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -o|--output)
                output_file="$2"
                shift 2
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            --)
                shift
                custom_args="$@"
                break
                ;;
            *)
                if [ -z "$task" ]; then
                    task="$1"
                fi
                shift
                ;;
        esac
    done
    
    # Validate inputs
    if [ -z "$task" ]; then
        echo -e "${RED}Error: Task description is required${NC}"
        show_help
        exit 1
    fi
    
    # Check if repomix is installed
    if ! command -v repomix &> /dev/null; then
        echo -e "${RED}Error: repomix is not installed${NC}"
        echo "Install with: npm install -g repomix"
        exit 1
    fi
    
    # Check if gemini is installed
    if ! command -v gemini &> /dev/null; then
        echo -e "${RED}Error: gemini CLI is not installed${NC}"
        echo "Install from: https://github.com/[gemini-cli-repo]"
        exit 1
    fi
    
    # Generate context file path
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local context_file="$TEMP_DIR/context_${mode}_${timestamp}.xml"
    
    echo -e "${GREEN}Planning with Gemini - Mode: $mode${NC}"
    echo -e "${GREEN}Task: $task${NC}"
    echo ""
    
    # Run repomix based on mode
    case $mode in
        auto)
            run_repomix_auto "$task" "$context_file"
            ;;
        full)
            run_repomix_full "$context_file"
            ;;
        quick)
            run_repomix_quick "$context_file"
            ;;
        security)
            run_repomix_security "$context_file"
            ;;
        custom)
            echo -e "${BLUE}Running custom repomix command...${NC}"
            repomix $custom_args --output "$context_file"
            ;;
    esac
    
    # Check if repomix succeeded
    if [ ! -f "$context_file" ]; then
        echo -e "${RED}Error: Failed to generate context with repomix${NC}"
        exit 1
    fi
    
    # Show context stats
    if [ "$verbose" = true ]; then
        echo -e "${YELLOW}Context file: $context_file${NC}"
        echo -e "${YELLOW}Size: $(du -h "$context_file" | cut -f1)${NC}"
    fi
    
    # Create Gemini prompt
    local prompt=$(create_gemini_prompt "$mode" "$task")
    
    # Run Gemini analysis
    run_gemini_analysis "$context_file" "$prompt" "$output_file"
    
    # Cleanup if not verbose
    if [ "$verbose" = false ]; then
        rm -f "$context_file"
    fi
    
    echo -e "${GREEN}Planning complete!${NC}"
}

# Run main function
main "$@"