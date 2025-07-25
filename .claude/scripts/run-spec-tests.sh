#!/bin/bash
# Run specification-based tests
# This script validates specifications and runs generated tests

set -e

echo "🧪 Specification-Based Testing Framework"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to validate a specification
validate_spec() {
    local spec_file="$1"
    echo -e "\n${YELLOW}Validating specification:${NC} $spec_file"
    
    if [ -f "$spec_file" ]; then
        python3 "$PROJECT_ROOT/utils/spec_validator.py" validate "$spec_file"
        return $?
    else
        echo -e "${RED}Specification not found:${NC} $spec_file"
        return 1
    fi
}

# Function to generate tests from specification
generate_tests() {
    local spec_file="$1"
    local output_dir="$2"
    
    echo -e "\n${YELLOW}Generating tests from:${NC} $spec_file"
    
    # Extract base name for test file
    base_name=$(basename "$spec_file" .md)
    test_file="$output_dir/test_${base_name}.py"
    
    python3 "$PROJECT_ROOT/utils/spec_validator.py" generate-tests "$spec_file" -o "$test_file"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Generated test file:${NC} $test_file"
        return 0
    else
        echo -e "${RED}Failed to generate tests${NC}"
        return 1
    fi
}

# Function to run tests
run_tests() {
    local test_dir="$1"
    
    echo -e "\n${YELLOW}Running tests in:${NC} $test_dir"
    
    if [ -d "$test_dir" ]; then
        # Check if pytest is installed
        if command -v pytest &> /dev/null; then
            pytest "$test_dir" -v
        else
            echo -e "${YELLOW}pytest not found, using unittest${NC}"
            python3 -m unittest discover "$test_dir" -v
        fi
    else
        echo -e "${RED}Test directory not found:${NC} $test_dir"
        return 1
    fi
}

# Main execution
case "${1:-all}" in
    validate)
        # Validate a specific specification
        if [ -z "$2" ]; then
            echo "Usage: $0 validate <spec_file>"
            exit 1
        fi
        validate_spec "$2"
        ;;
        
    generate)
        # Generate tests from specification
        if [ -z "$2" ]; then
            echo "Usage: $0 generate <spec_file> [output_dir]"
            exit 1
        fi
        output_dir="${3:-$PROJECT_ROOT/tests}"
        mkdir -p "$output_dir"
        generate_tests "$2" "$output_dir"
        ;;
        
    test)
        # Run tests
        test_dir="${2:-$PROJECT_ROOT/tests}"
        run_tests "$test_dir"
        ;;
        
    all)
        # Full workflow: validate all specs, generate tests, run tests
        echo "Running full specification-based testing workflow..."
        
        # Create directories if they don't exist
        mkdir -p "$PROJECT_ROOT/docs/specs"
        mkdir -p "$PROJECT_ROOT/tests"
        
        # Find all specification files
        specs_found=false
        for spec in "$PROJECT_ROOT/docs/specs"/SPEC-*.md; do
            if [ -f "$spec" ]; then
                specs_found=true
                
                # Validate specification
                if validate_spec "$spec"; then
                    # Generate tests if validation passes
                    generate_tests "$spec" "$PROJECT_ROOT/tests"
                fi
            fi
        done
        
        if [ "$specs_found" = false ]; then
            echo -e "\n${YELLOW}No specifications found in docs/specs/${NC}"
            echo "Create specifications using the specification template."
            exit 0
        fi
        
        # Run all tests
        echo -e "\n${GREEN}Running all generated tests...${NC}"
        run_tests "$PROJECT_ROOT/tests"
        ;;
        
    *)
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  validate <spec_file>    Validate a specification"
        echo "  generate <spec_file>    Generate tests from specification"
        echo "  test [test_dir]         Run tests"
        echo "  all                     Run full workflow (default)"
        echo ""
        echo "Examples:"
        echo "  $0 validate docs/specs/SPEC-2024-01-01-feature.md"
        echo "  $0 generate docs/specs/SPEC-2024-01-01-feature.md"
        echo "  $0 test"
        echo "  $0 all"
        exit 1
        ;;
esac