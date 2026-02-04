#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict

# ANSI
CLR = "\033[2J\033[H"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[31m"
YEL = "\033[33m"
BLU = "\033[34m"
GRN = "\033[32m"
CYAN = "\033[36m"

SEV_ORDER = {"error": 0, "warning": 1, "information": 2}
SEV_ICON = {"error": "✖", "warning": "▲", "information": "●"}
SEV_COLOR = {"error": RED, "warning": YEL, "information": BLU}

def term_width(default=100):
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default

def hr(ch="─"):
    return ch * term_width()

def shorten(s: str, max_len: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)] + "…"

def read_json_objects(stream):
    buf = ""
    depth = 0
    in_str = False
    esc = False
    started = False

    while True:
        ch = stream.read(1)
        if not ch:
            return

        buf += ch

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1

        if started and depth == 0:
            s = buf.strip()
            buf = ""
            started = False
            if not s:
                continue
            yield json.loads(s)

def fmt_loc(d):
    r = d.get("range") or {}
    s = r.get("start") or {}
    # basedpyright is 0-based
    line = (s.get("line", 0) or 0) + 1
    col = (s.get("character", 0) or 0) + 1
    return line, col

def colorize_sev(sev: str, text: str) -> str:
    return f"{SEV_COLOR.get(sev, '')}{text}{RESET}"

def main():
    for obj in read_json_objects(sys.stdin):
        diags = obj.get("generalDiagnostics") or []
        summ = obj.get("summary") or {}
        analyzed = summ.get("filesAnalyzed", "?")
        ec = summ.get("errorCount", 0)
        wc = summ.get("warningCount", 0)
        ic = summ.get("informationCount", 0)
        tsec = summ.get("timeInSec", "?")

        # group by file
        by_file = defaultdict(list)
        for d in diags:
            by_file[d.get("file", "<unknown>")].append(d)

        # header
        parts = []
        parts.append(f"{BOLD}basedpyright{RESET} {DIM}(watch){RESET}")
        parts.append(f"files {BOLD}{analyzed}{RESET}")
        parts.append(f"{colorize_sev('error', 'errors')} {BOLD}{ec}{RESET}")
        parts.append(f"{colorize_sev('warning', 'warnings')} {BOLD}{wc}{RESET}")
        parts.append(f"{colorize_sev('information', 'info')} {BOLD}{ic}{RESET}")
        parts.append(f"time {BOLD}{tsec}{RESET}s")
        header = "  ".join(parts)

        out = []
        out.append(header)
        out.append(hr())

        if not diags:
            out.append(f"{GRN}✓ No diagnostics.{RESET}")
        else:
            # sort files by "worst" severity then name
            def file_key(path):
                worst = min(SEV_ORDER.get(d.get("severity", "warning"), 9) for d in by_file[path])
                return (worst, path)

            for f in sorted(by_file.keys(), key=file_key):
                items = sorted(
                    by_file[f],
                    key=lambda d: (SEV_ORDER.get(d.get("severity", "warning"), 9), *fmt_loc(d)),
                )

                # file summary counts
                counts = defaultdict(int)
                for d in items:
                    counts[d.get("severity", "warning")] += 1

                badges = []
                if counts["error"]:
                    badges.append(colorize_sev("error", f"{SEV_ICON['error']} {counts['error']}"))
                if counts["warning"]:
                    badges.append(colorize_sev("warning", f"{SEV_ICON['warning']} {counts['warning']}"))
                if counts["information"]:
                    badges.append(colorize_sev("information", f"{SEV_ICON['information']} {counts['information']}"))

                out.append(f"\n{BOLD}{f}{RESET}  {DIM}{'  '.join(badges)}{RESET}")

                # diagnostics list
                w = term_width()
                msg_max = max(30, w - 28)  # heuristic for line wrapping avoidance
                for d in items:
                    sev = d.get("severity", "warning")
                    icon = SEV_ICON.get(sev, "•")
                    rule = d.get("rule")
                    line, col = fmt_loc(d)
                    loc = f"{line}:{col}"
                    rule_part = f"{DIM}[{rule}]{RESET}" if rule else ""
                    msg = shorten(d.get("message", ""), msg_max)

                    sev_tag = colorize_sev(sev, f"{icon} {sev.upper():<7}")
                    out.append(f"  {sev_tag} {DIM}{loc:<9}{RESET} {msg} {rule_part}".rstrip())

        sys.stdout.write(CLR + "\n".join(out) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
