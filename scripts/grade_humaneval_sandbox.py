#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit-test executor for HumanEval-164 traces (isolated `python -I -B` subprocess,
4.0 s timeout per item, 8 parallel workers).

SECURITY WARNING: this script executes model-generated Python code with the
privileges of the current user. The `-I` flag isolates the interpreter
environment only; it is NOT an OS-level sandbox. Run it inside a disposable
container or VM (e.g. `docker run --rm --network none ...`).

Note: TIMEOUT counts depend on CPU load and may shift by +-1 item between
machines; PASS counts are unaffected across verified test runs.

Run from repository root:
    python3 scripts/grade_humaneval_sandbox.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import concurrent.futures
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_PATH = os.path.join(REPO_ROOT, "suites", "suite_unseen_humaneval164.json")

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
TOP_IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+import\s+|import\s+)", re.MULTILINE)


def exact_mcnemar_p(gains: int, losses: int) -> float:
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, (2 * tail) / (1 << n))


def extract_candidate_code(after_think: str, entry_point: str, sig_prompt: str) -> str | None:
    clean = after_think.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if not clean:
        return None
    header_lines = [
        ln for ln in sig_prompt.splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    blocks = FENCE_RE.findall(clean)
    if blocks:
        for b in reversed(blocks):
            if f"def {entry_point}" in b:
                missing_hdr = [ln for ln in header_lines if ln.strip() not in b]
                return ("\n".join(missing_hdr) + "\n\n" + b) if missing_hdr else b
        last_b = blocks[-1]
        if "def " not in last_b:
            return sig_prompt + "\n" + last_b
        return last_b
    def_pos = clean.find(f"def {entry_point}")
    if def_pos >= 0:
        m = TOP_IMPORT_RE.search(clean)
        idx = m.start() if (m and m.start() < def_pos) else def_pos
        return clean[idx:]
    return None


def eval_one_item(args: tuple[dict, dict]) -> dict:
    task, rec = args
    uid = task["unique_id"]
    entry = task["entry_point"]
    sig = task["code_signature_prompt"]
    test_code = task["test_code"]
    thinking_on = bool(rec.get("thinking_on", True))
    txt = rec.get("text", "")
    hit_eos = bool(rec.get("hit_eos", False))

    if thinking_on:
        if "</think>" not in txt:
            return {
                "unique_id": uid,
                "passed": False,
                "status": "UNCLOSED_THINK_32K",
                "hit_eos": hit_eos,
                "total_tokens": rec.get("total_tokens", 0),
                "think_tokens": rec.get("think_tokens", 0),
                "error_detail": "Model never closed </think> within 32,768 tokens",
            }
        after = txt.rsplit("</think>", 1)[1]
    else:
        after = txt

    code = extract_candidate_code(after, entry, sig)
    if not code:
        return {
            "unique_id": uid,
            "passed": False,
            "status": "NO_CODE_EXTRACTED",
            "hit_eos": hit_eos,
            "total_tokens": rec.get("total_tokens", 0),
            "think_tokens": rec.get("think_tokens", 0),
            "error_detail": "No Python code block found after </think>",
        }

    full_prog = f"{code}\n\n{test_code}\n\ncheck({entry})\nprint('__HUMANEVAL_ALL_TESTS_PASSED__')\n"
    with tempfile.TemporaryDirectory() as tmpd:
        script_p = os.path.join(tmpd, "sol.py")
        with open(script_p, "w", encoding="utf-8") as sf:
            sf.write(full_prog)
        try:
            cp = subprocess.run(
                [sys.executable, "-I", "-B", script_p],
                cwd=tmpd,
                capture_output=True,
                text=True,
                timeout=4.0,
            )
        except subprocess.TimeoutExpired:
            return {
                "unique_id": uid,
                "passed": False,
                "status": "TIMEOUT_INFINITE_LOOP",
                "hit_eos": hit_eos,
                "total_tokens": rec.get("total_tokens", 0),
                "think_tokens": rec.get("think_tokens", 0),
                "error_detail": "Subprocess timed out (>4.0s)",
            }

    if cp.returncode == 0 and "__HUMANEVAL_ALL_TESTS_PASSED__" in cp.stdout:
        return {
            "unique_id": uid,
            "passed": True,
            "status": "PASS",
            "hit_eos": hit_eos,
            "total_tokens": rec.get("total_tokens", 0),
            "think_tokens": rec.get("think_tokens", 0),
            "error_detail": "",
        }
    err_tail = (cp.stderr or cp.stdout).strip().splitlines()
    last_err = err_tail[-1][:200] if err_tail else f"rc={cp.returncode}"
    if "SyntaxError" in (cp.stderr or "") or "IndentationError" in (cp.stderr or ""):
        st = "SYNTAX_ERROR"
    elif "AssertionError" in (cp.stderr or ""):
        st = "ASSERTION_FAIL"
    else:
        st = "RUNTIME_ERROR"
    return {
        "unique_id": uid,
        "passed": False,
        "status": st,
        "hit_eos": hit_eos,
        "total_tokens": rec.get("total_tokens", 0),
        "think_tokens": rec.get("think_tokens", 0),
        "error_detail": last_err,
    }


def grade_trace_file(jsonl_path: str, suite_by_id: dict[str, dict], write_json: bool = False) -> dict:
    recs = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                recs[r["unique_id"]] = r
    pairs = [(suite_by_id[uid], recs[uid]) for uid in sorted(suite_by_id.keys()) if uid in recs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        graded = list(ex.map(eval_one_item, pairs))

    n_pass = sum(1 for g in graded if g["passed"])
    n_tot = len(graded)
    n_eos = sum(1 for g in graded if g["hit_eos"])
    mean_tok = sum(g["total_tokens"] for g in graded) / max(n_tot, 1)
    mean_think = sum(g["think_tokens"] for g in graded) / max(n_tot, 1)
    by_status: dict[str, int] = {}
    for g in graded:
        by_status[g["status"]] = by_status.get(g["status"], 0) + 1

    with open(jsonl_path, "rb") as bf:
        sha = hashlib.sha256(bf.read()).hexdigest()
    model_id = next(iter(recs.values()))["model_id"] if recs else os.path.basename(jsonl_path)
    out = {
        "trace_file": os.path.basename(jsonl_path),
        "trace_sha256": sha,
        "model_id": model_id,
        "n_evaluated": n_tot,
        "pass_at_1_count": n_pass,
        "pass_at_1_pct": round(100.0 * n_pass / max(n_tot, 1), 2),
        "eos_pct": round(100.0 * n_eos / max(n_tot, 1), 2),
        "cut_32k_pct": round(100.0 * (n_tot - n_eos) / max(n_tot, 1), 2),
        "mean_tokens": round(mean_tok, 1),
        "mean_think_tokens": round(mean_think, 1),
        "status_breakdown": by_status,
        "per_problem": graded,
    }
    if write_json:
        out_p = jsonl_path.replace(".jsonl", "_humaneval_exec.json")
        with open(out_p, "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False, indent=2)
    print(
        f"[HumanEval Sandbox] {os.path.basename(jsonl_path):45s} -> "
        f"pass@1={n_pass:3d}/{n_tot:3d} ({out['pass_at_1_pct']:5.2f}%) | "
        f"cut={out['cut_32k_pct']:5.2f}% | tok={mean_tok:7.1f} (think={mean_think:7.1f}) | {by_status}"
    )
    return out


def main() -> None:
    write_json = "--write-json" in sys.argv
    cli_paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    with open(SUITE_PATH, encoding="utf-8") as sf:
        suite = json.load(sf)
    suite_by_id = {t["unique_id"]: t for t in suite}
    paths = cli_paths if cli_paths else sorted(glob.glob(os.path.join(REPO_ROOT, "traces", "humaneval164_*.jsonl")))
    pass_maps: dict[str, dict[str, bool]] = {}
    think_means: dict[str, float] = {}
    for p in paths:
        res = grade_trace_file(p, suite_by_id, write_json=write_json)
        pass_maps[res["model_id"]] = {g["unique_id"]: bool(g["passed"]) for g in res["per_problem"]}
        think_means[res["model_id"]] = res["mean_think_tokens"]

    if "SGNE_CC" in pass_maps and "MIT_p50c" in pass_maps:
        sg, mt = pass_maps["SGNE_CC"], pass_maps["MIT_p50c"]
        g = sum(1 for k in sg if sg[k] and not mt[k])
        l = sum(1 for k in sg if not sg[k] and mt[k])
        ratio = think_means["MIT_p50c"] / max(think_means["SGNE_CC"], 1e-9)
        print(f"  McNemar [SGNE-CC vs MIT-p50c]: +{g}/-{l}, exact two-sided p = {exact_mcnemar_p(g, l):.4f} | <think> compression = {ratio:.2f}x")
    if "SGNE_CC" in pass_maps and "FACTORY_BASE" in pass_maps:
        sg, bs = pass_maps["SGNE_CC"], pass_maps["FACTORY_BASE"]
        g = sum(1 for k in sg if sg[k] and not bs[k])
        l = sum(1 for k in sg if not sg[k] and bs[k])
        print(f"  McNemar [SGNE-CC vs Base]    : +{g}/-{l}, exact two-sided p = {exact_mcnemar_p(g, l):.3e}")


if __name__ == "__main__":
    main()
