#!/usr/bin/env python3
"""Enhanced pre-tool-use hook for AI safety research"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime

# Dangerous command patterns
DANGEROUS_COMMANDS = [
    r"\brm\s+.*-[a-z]*r[a-z]*f",
    r"\brm\s+.*-[a-z]*f[a-z]*r",
    r"\bdd\s+if=/dev/zero",
    r"\b:\(\)\{\:\|\:&\};\:",
]

SENSITIVE_FILES = [
    r"\.env$",
    r"secrets?\.",
    r"\.pem$",
    r"\.key$",
    r"id_rsa",
]


def is_dangerous_command(command):
    normalized = " ".join(command.lower().split())
    for pattern in DANGEROUS_COMMANDS:
        if re.search(pattern, normalized):
            return True
    return False


def is_sensitive_file(file_path):
    for pattern in SENSITIVE_FILES:
        if re.search(pattern, file_path, re.IGNORECASE):
            if ".env.sample" in file_path or ".env.example" in file_path:
                continue
            return True
    return False


def main():
    try:
        input_data = json.load(sys.stdin)
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Check dangerous commands
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            if is_dangerous_command(command):
                print("BLOCKED: Dangerous command detected", file=sys.stderr)
                sys.exit(2)

        # Check sensitive files
        if tool_name in ["Read", "Edit", "Write"]:
            file_path = tool_input.get("file_path", "")
            if is_sensitive_file(file_path):
                print("BLOCKED: Access to sensitive file", file=sys.stderr)
                sys.exit(2)

        # Log activity
        log_dir = Path.cwd() / ".claude" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON log
        log_path = log_dir / "pre_tool_use.json"
        log_data = []
        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    log_data = json.load(f)
                except:
                    pass

        log_entry = input_data.copy()
        log_entry["timestamp"] = datetime.now().isoformat()
        log_data.append(log_entry)

        # Keep last 1000 entries
        log_data = log_data[-1000:]

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        sys.exit(0)

    except:
        sys.exit(0)


if __name__ == "__main__":
    main()
