"""Strip host / infrastructure identifiers from stored experiment artefacts.

    python3 scripts/sanitize_artifacts.py [--check]

Run artefacts written before ``lorr_job.py`` / ``ray_sweep.py`` stopped
recording infrastructure details carried a hostname, scratch paths, and per-node
job counts keyed by IP address.  This script rewrites them in place:

* ``summary.json``      drop ``host`` / ``node_ip``; make ``cmd`` path-free
* ``_manifest_*.json``  drop ``jobs_per_node`` and ``max_concurrent``
* ``*.log``             drop Ray connection / warning lines; mask IP addresses
* any other text file   replace absolute paths under the project root with ``.``

``--check`` only reports remaining matches (exit code 1 if any), so it can be
used as a pre-release gate.  Scientific content (metrics, configs) is untouched.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, "results")

IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
SCRATCH = re.compile(r"(?:/[\w.\-]+)*/lorr_jobs/[\w.\-]+/[\w.\-]+/")
# Patterns that must not survive in released text files.  Site-specific
# identifiers are derived at run time (project path, its mount root, this
# machine's hostname) or passed via SANITIZE_EXTRA_PATTERNS (one regex; use |
# for alternatives), so this file itself names no infrastructure.
_root = os.sep + PROJECT.strip(os.sep).split(os.sep)[0] if PROJECT.strip(os.sep) else None
FORBIDDEN = [re.compile(p) for p in (
    re.escape(PROJECT), *([re.escape(_root) + r"\b"] if _root and _root not in ("/home", "/Users", "/tmp") else []),
    re.escape(socket.gethostname()), r"\"host\":", r"jobs_per_node", r"node_ip",
    *[x for x in [os.environ.get("SANITIZE_EXTRA_PATTERNS", "")] if x],
)]
LOG_DROP = re.compile(r"Connecting to existing Ray cluster|Connected to Ray cluster|View the dashboard|"
                      r"FutureWarning|warnings\.warn|RAY_ACCEL_ENV_VAR")
TEXT_EXT = {".json", ".csv", ".log", ".md", ".txt", ".tex", ".html", ".py", ".sh", ".bib"}


def clean_cmd(cmd: str) -> str:
    cmd = SCRATCH.sub("", cmd)
    return cmd.replace(PROJECT + os.sep, "")


def sanitize_summary(path):
    with open(path) as f:
        s = json.load(f)
    before = json.dumps(s, sort_keys=True)
    s.pop("host", None)
    s.pop("node_ip", None)
    if "cmd" in s:
        s["cmd"] = clean_cmd(s["cmd"])
    if json.dumps(s, sort_keys=True) != before:
        with open(path, "w") as f:
            json.dump(s, f)
        return 1
    return 0


def sanitize_manifest(path):
    with open(path) as f:
        m = json.load(f)
    changed = False
    for k in ("jobs_per_node", "max_concurrent", "platform", "host"):
        if k in m:
            m.pop(k)
            changed = True
    if changed:
        with open(path, "w") as f:
            json.dump(m, f, indent=1)
    return int(changed)


def sanitize_log(path):
    with open(path, errors="replace") as f:
        lines = f.readlines()
    out = []
    for ln in lines:
        if LOG_DROP.search(ln):
            continue
        ln = IP.sub("<redacted>", ln).replace(PROJECT + os.sep, "").replace(PROJECT, ".")
        out.append(ln)
    text = "".join(out)
    # the sweep driver echoed its manifest; drop the per-node block and concurrency knob
    text = re.sub(r'\s*"jobs_per_node": \{[^}]*\},?', "", text)
    text = re.sub(r'\s*"max_concurrent": \d+,?', "", text)
    if text != "".join(lines):
        with open(path, "w") as f:
            f.write(text)
        return 1
    return 0


def sanitize_text(path):
    with open(path, errors="replace") as f:
        s = f.read()
    t = s.replace(PROJECT + os.sep, "").replace(PROJECT, ".")
    if t != s:
        with open(path, "w") as f:
            f.write(t)
        return 1
    return 0


def text_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__", "build", "Benchmark-Archive")]
        for fn in fns:
            if os.path.splitext(fn)[1] in TEXT_EXT:
                yield os.path.join(dp, fn)


def check():
    hits = []
    for p in text_files(PROJECT):
        if os.path.abspath(p) == os.path.abspath(__file__):
            continue
        with open(p, errors="replace") as f:
            s = f.read()
        for rx in FORBIDDEN:
            m = rx.search(s)
            if m:
                hits.append((os.path.relpath(p, PROJECT), m.group(0)))
                break
        if p.endswith(".log") and IP.search(s):
            hits.append((os.path.relpath(p, PROJECT), "ip address"))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        n = {"summary": 0, "manifest": 0, "log": 0, "text": 0}
        for p in glob.glob(os.path.join(RESULTS, "**", "summary.json"), recursive=True):
            n["summary"] += sanitize_summary(p)
        for p in glob.glob(os.path.join(RESULTS, "**", "*manifest*.json"), recursive=True):
            n["manifest"] += sanitize_manifest(p)
        for p in glob.glob(os.path.join(RESULTS, "**", "*.log"), recursive=True):
            n["log"] += sanitize_log(p)
        for p in text_files(PROJECT):
            if os.path.abspath(p) != os.path.abspath(__file__):
                n["text"] += sanitize_text(p)
        print("rewritten:", n)
    hits = check()
    for h in hits[:50]:
        print("REMAINING:", *h)
    print(f"{len(hits)} file(s) with forbidden patterns")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
