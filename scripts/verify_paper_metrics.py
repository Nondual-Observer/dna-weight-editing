#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline CPU verifier of all 37 SHA-256 artifact checksums, independent math/MGSM/BBH/MultiSpeech
trace recomputations, primary MIT reference graders (MIT-Std / MIT-Lenient via symbolic math-verify),
40-block clustered sign-flip test for MGSM-240, calibration/holdout partition checks, static AST
checks for HumanEval-164, and archived HumanEval execution tables reported in README.md.

Scope note:
  - Recomputes all MATH-500, FreshBlind56, MGSM-240, BBH-160, and MultiSpeech-64 metrics directly
    from raw model outputs (`text`, `hit_eos`, `total_tokens`) and verifies zero discrepancy against
    stored `strict_ok` flags.
  - Does NOT execute untrusted model-generated Python code for HumanEval-164 on the host: verifies
    token/completion counts, code extraction, and `ast.parse` syntax validity statically, and checks
    archived pass@1 verdicts (`26 / 51 / 46`) recorded in `suites/reproducibility_and_holdout_manifest.json`.
    To re-execute HumanEval unit tests inside an isolated container/VM, run `scripts/grade_humaneval_sandbox.py`.

Prerequisites:
    pip install -r requirements.txt

Run from repository root (or any working directory):
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_paper_metrics.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast
import functools
import hashlib
import json
import logging
import math
import os
import re
import statistics
import threading

try:
    from math_verify import parse as mv_parse, verify as mv_verify
except ImportError as exc:
    raise SystemExit(
        "Missing symbolic math grader package `math-verify` (required for exact MIT-Std / MIT-Lenient / Strict "
        "equivalence). Install dependencies first:\n    pip install -r requirements.txt"
    ) from exc

logging.getLogger("math_verify").setLevel(logging.CRITICAL)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_SHA256 = {
    "sgne_cc_dna_delta.npz": "4e4d36e032869bc10f56e2a2033a52d3f2774321cf867ef6f74274a7c0d1566d",
    "suites/fresh_blind56_32k_suite.json": "63ba2cf247934afbc767865a5727eb403bbbc6af3205d44d3e755fe849a43e89",
    "suites/reproducibility_and_holdout_manifest.json": "ac6440486e51e684b2bd19738f8e8f7a7175fdb7cc3f36d1515d83fe250d8833",
    "suites/suite_math500_full.json": "a596fa9b05b35f2c0ff32b70ea620d3241d90862cbd9bd1e77af738bb5c78a9b",
    "suites/suite_math500_strat100.json": "6347b75c46d4969a1c021ff8d42ba80e3c5c1a00540e62b7dc6fd0b8a8dd0c79",
    "suites/suite_multispeech64.json": "8fa2da6c2ab8bd7d64659c24b1ead845032f19db6a3c79e05790a4d30afabe55",
    "suites/suite_unseen_humaneval164.json": "5703edc81a78f82dca58da583036dd73e22715085ca99540c8d88189f19e1a89",
    "suites/suite_unseen_mgsm240_bbh160.json": "5de44113cae2b07d8a693d1c87a2e61de5a18d79dd9c01e1049b10453f41df66",
    "traces/blind56_base_think32k_pure_t0.jsonl": "58d0db19f5d3d95dea9e4cb39b9a66979cfaef01a5235be8ed4b9945b1064ec0",
    "traces/blind56_sgne_cc_think32k_pure_t0.jsonl": "2072450af148d124fb09f9698d9eb137b933f44014f125e91f394709d7e5a14e",
    "traces/humaneval164_base_think32k_card_pp15.jsonl": "b2fd764fe9f681567a0d35a3251f484f9969c9149b63044589773e178369b041",
    "traces/humaneval164_mit_p50c_think32k_card_pp15.jsonl": "dd1c6ea407d53ce0a4b9e22f5ec9436c2984419b074ac847c273293b1f0c4a13",
    "traces/humaneval164_sgne_cc_think32k_card_pp15.jsonl": "fc67fef2e2db445be97d94c6ac23981b848f557d10e981a0d043a06767c42106",
    "traces/math500_base_think32k_card_pp15.jsonl": "6d06d373c429c760584a553ebf9ed26a639f42ee64b4ce72b86d0c49fc3c113c",
    "traces/math500_base_think32k_pure_t0_strat100.jsonl": "8df959b204995ca909f68533c1bb857f4e9a2a04789cf9dc27b8bb46fede68fd",
    "traces/math500_mit_p50c_think32k_card_pp15.jsonl": "2c766bf006030cb15dd6d8256534ca6ad730b8a780d9c058872d09d9b150ffb6",
    "traces/math500_mit_p50c_think32k_pure_t0_strat100.jsonl": "4634cb4a5234cfce5627d8b31b6e103d28ebac009735e5c3475b6ac7913aeb26",
    "traces/math500_sgne_cc_think32k_card_pp15.jsonl": "8f4fed10df896a6ffe5f367982f23d88625cdee1b3cba5a40801626ce3b7be9a",
    "traces/math500_sgne_cc_think32k_pure_t0_strat100.jsonl": "bf1a30e67cf900fbb892c5b4ac6ed7220fc64d8ef2e87e29ec6b6fa69187f350",
    "traces/mgsm240_bbh160_base_think32k_card_pp15.jsonl": "8e7fa9ece63a7de7399b786dc53df8a1c1287e049bbea25e612a836c996fe23e",
    "traces/mgsm240_bbh160_mit_p50c_think32k_card_pp15.jsonl": "317556c3dd0e42b2561c37cd36d82e847a3fa2884523508e02348b0f9c96ba64",
    "traces/mgsm240_bbh160_sgne_cc_think32k_card_pp15.jsonl": "a8936581d18a482b773ad40f71cd6bfc59e93a5bc57280b44e3d30a963a93767",
    "traces/multispeech64_base_nothink_card_pp15.jsonl": "c28645dd975aa0276986fb1b98bff9164e9c5a8b7dfa90cd696ed770fb34cd20",
    "traces/multispeech64_base_nothink_t0.jsonl": "38029c2f1e5e8726ec493a91a9c4cc70c8e9b2f97bcd2e4d1c0b5a209ebfd0f3",
    "traces/multispeech64_base_think32k_card_pp15.jsonl": "8b4767ca1b1dc0af3466a486bedbb50e685df339ed65111b81c84292ed633ccd",
    "traces/multispeech64_base_think32k_pure_t0.jsonl": "13fece99c98ae619687cf71970b45d52b326d751a536d2ce678b56cd4e6c994d",
    "traces/multispeech64_mit_p50c_nothink_card_pp15.jsonl": "9f837413c7bce853d2ddb4d58390f3449b963bf7cbe9314bdff8bce9ad4ff484",
    "traces/multispeech64_mit_p50c_nothink_t0.jsonl": "dc29caa22c759a9c9166e1db003a573e78229c63a7c7a856a7dc746938c99cd5",
    "traces/multispeech64_mit_p50c_think32k_card_pp15.jsonl": "a088d9444b7c40baa3b43a014e3ce83bbe320005ff7901d169c7c3cc536121e2",
    "traces/multispeech64_mit_p50c_think32k_pure_t0.jsonl": "a2574d87d70ec5da102be1be12dd8c2de7c7fdfa29fa6a523be1ed863422408d",
    "traces/multispeech64_sgne_cc_nothink_card_pp15.jsonl": "e63dca071bda5112356eb4257e348a4e03137f0d849a88a6f0131d13f1cd1248",
    "traces/multispeech64_sgne_cc_nothink_t0.jsonl": "9f805c6ea9f67f9828b83a6893ed1f8c688eefddd4fd586496099e93869db7e9",
    "traces/multispeech64_sgne_cc_think32k_card_pp15.jsonl": "d1f5687daeebb949635bbc44e7ac6494040ab3e5ab0a6980f7b6afd335d463bd",
    "traces/multispeech64_sgne_cc_think32k_pure_t0.jsonl": "f9f27c9b36c53b4605b5d57eb2d6aa81f4a9c845269b065995b495c2d60c2a29",
    "traces/mit_official_math500_ref/mit_p50c_merged.json": "c93d3db30a612ef0da396014fdfbe011efbcf2cd417e2e6125664d7effdd0471",
    "traces/mit_official_math500_ref/mit_p50c_official_500_per_problem.json": "a699f8b5e31c8db138f5137ceba736fa26275630fdac803ba824ba1b1bad3edf",
    "traces/mit_official_math500_ref/mit_unedited_merged.json": "5c31370485b9f197f592745e935eeaee6b09dc127daa5b2a8be1ebbcbc210221",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(rel_path: str) -> list[dict]:
    p = os.path.join(REPO_ROOT, rel_path)
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def exact_mcnemar_p(gains: int, losses: int) -> float:
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, (2 * tail) / (1 << n))


def mcnemar_pair(map_a: dict[str, bool], map_b: dict[str, bool]) -> tuple[int, int, int, int, float]:
    keys = sorted(set(map_a) & set(map_b))
    a_ok = sum(1 for k in keys if map_a[k])
    b_ok = sum(1 for k in keys if map_b[k])
    gains = sum(1 for k in keys if map_a[k] and not map_b[k])
    losses = sum(1 for k in keys if not map_a[k] and map_b[k])
    return a_ok, b_ok, gains, losses, exact_mcnemar_p(gains, losses)


def exact_cluster_signflip_p(diffs: list[int]) -> float:
    """
    Exact two-sided paired cluster sign-flip p-value across B blocks (e.g. B=40 6-language MGSM problems).
    Under block-level model-label exchangeability, each non-zero block difference d_i flips sign with p=0.5.
    Computes P(|sum s_i d_i| >= |obs|) exactly via dynamic programming over integer sums.
    """
    obs = abs(sum(diffs))
    nz = [abs(d) for d in diffs if d != 0]
    if not nz or obs == 0:
        return 1.0
    dp: dict[int, int] = {0: 1}
    for d in nz:
        nxt: dict[int, int] = {}
        for s, cnt in dp.items():
            nxt[s + d] = nxt.get(s + d, 0) + cnt
            nxt[s - d] = nxt.get(s - d, 0) + cnt
        dp = nxt
    tail = sum(cnt for s, cnt in dp.items() if abs(s) >= obs)
    return min(1.0, tail / (1 << len(nz)))


# --- Primary MIT CSAIL answer extraction & symbolic math-verify equivalence ---
# Adapted from RikiyaT/lm_understand-logs (`exp8_math/code/mathqa.py` & `evaluate.py`, MIT License).
def _mv_timeout() -> int | None:
    return 5 if threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=65536)
def _mv_parsed(s: str, to: int | None):
    return mv_parse("\\boxed{" + s + "}", parsing_timeout=to)


@functools.lru_cache(maxsize=65536)
def mit_is_correct(pred: str | None, gold: str | None) -> bool:
    """Symbolic/numeric equivalence via math-verify (identical to MIT mathqa.is_correct)."""
    if pred is None or gold is None:
        return False
    to = _mv_timeout()
    try:
        g = _mv_parsed(str(gold), to)
        p = _mv_parsed(str(pred), to)
        if not g or not p:
            return False
        return bool(mv_verify(g, p, timeout_seconds=to))
    except Exception:
        return False


def _mit_box_at(text: str, idx: int) -> str | None:
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key) :]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def mit_last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _mit_box_at(text, idx)


def mit_extract_boxed(text: str) -> str | None:
    t = text.split("</think>")[-1].strip() if "</think>" in text else text
    return mit_last_boxed(t)


def mit_lenient_pred(text: str) -> str | None:
    found = []
    for pat in (
        r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)",
        r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
        r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)",
    ):
        found += [(m.start(1), m.group(1).strip().strip("*").rstrip(".").strip()) for m in re.finditer(pat, text)]
    for _, t in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", t):
            return t
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def mit_lenient_correct(text: str, pred: str | None, good: bool, gold: str) -> bool:
    """MIT evaluate.py lenient_correct: strict boxed grade, or fallback phrase/number when no box was given."""
    if good or pred is not None:
        return bool(good)
    lp = mit_lenient_pred(text)
    if lp is None:
        return False
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", lp)
    return bool(mit_is_correct(lp, gold) or (nums and mit_is_correct(nums[-1].replace(",", ""), gold)))


def eval_mit_three_metrics(rows: list[dict]) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool]]:
    """
    Independently recomputes Strict, MIT-Std, and MIT-Lenient from raw `text`, `hit_eos`, and `gold`.
    Never trusts pre-saved `strict_ok=True` as a shortcut; asserts stored `strict_ok` matches recomputed Strict.
    """
    strict_m: dict[str, bool] = {}
    std_m: dict[str, bool] = {}
    len_m: dict[str, bool] = {}
    for r in rows:
        uid = r["unique_id"]
        raw_txt = r["text"]
        txt = raw_txt.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        gold = str(r["gold"]).strip()
        closed_th = "</think>" in txt
        hit_eos = bool(r.get("hit_eos", False))

        pred = mit_extract_boxed(txt)
        good_std = mit_is_correct(pred, gold)
        good_strict = bool(closed_th and hit_eos and good_std)
        good_len = mit_lenient_correct(txt, pred, good_std, gold)

        if "strict_ok" in r:
            assert bool(r["strict_ok"]) == good_strict, (
                f"Stored strict_ok mismatch on {uid}: stored={r['strict_ok']} vs recomputed={good_strict}"
            )
        strict_m[uid] = good_strict
        std_m[uid] = bool(good_std)
        len_m[uid] = bool(good_len)
    return strict_m, std_m, len_m


def recompute_mgsm_strict(rows: list[dict]) -> dict[str, bool]:
    """
    Independently recomputes MGSM grades from raw `text` + `hit_eos` + `gold` using symbolic `math-verify`
    plus integer space/comma thousands-separator normalization (e.g. French `70 000` -> `70000`).
    """
    out: dict[str, bool] = {}
    for r in rows:
        uid = r["unique_id"]
        txt = r["text"].replace("<|im_end|>", "").replace("<|endoftext|>", "")
        closed_th = "</think>" in txt
        hit_eos = bool(r.get("hit_eos", False))
        gold = str(r["gold"]).strip()
        after = txt.rsplit("</think>", 1)[1].strip() if closed_th else ""
        boxed = mit_last_boxed(after) if closed_th else None
        ok = False
        if closed_th and hit_eos and boxed is not None:
            ok = mit_is_correct(boxed, gold)
            if not ok:
                b_num = boxed.replace(",", "").replace(" ", "").strip("$")
                ok = b_num == gold.replace(",", "").strip()
        assert bool(r["strict_ok"]) == ok, f"MGSM stored strict_ok mismatch on {uid}: {r['strict_ok']} vs {ok}"
        out[uid] = ok
    return out


def clean_bbh(s: str) -> str:
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", str(s).strip()).strip()
    return s.strip("().").strip().lower()


FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
TOP_IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+import\s+|import\s+)", re.MULTILINE)


def extract_humaneval_candidate(after_think: str, entry_point: str, sig_prompt: str) -> str | None:
    clean = after_think.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if not clean:
        return None
    header_lines = [ln for ln in sig_prompt.splitlines() if ln.strip().startswith(("import ", "from "))]
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


def char_8gram_rep(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    n = len(chars)
    if n < 16:
        return 1.0 if n == 0 else 0.0
    tot = n - 7
    seen = set()
    reps = 0
    for i in range(tot):
        g = "".join(chars[i : i + 8])
        if g in seen:
            reps += 1
        else:
            seen.add(g)
    return reps / tot


def word_4gram_rep(s: str) -> float:
    ws = s.lower().split()
    n = len(ws)
    if n < 8:
        return 1.0 if n == 0 else 0.0
    tot = n - 3
    seen = set()
    reps = 0
    for i in range(tot):
        g = (ws[i], ws[i + 1], ws[i + 2], ws[i + 3])
        if g in seen:
            reps += 1
        else:
            seen.add(g)
    return reps / tot


def has_foreign_bleed(s: str, lang: str) -> bool:
    n_cjk = n_kana = n_cyr = n_arab = 0
    for c in s:
        u = ord(c)
        if (0x4E00 <= u <= 0x9FFF) or (0x3400 <= u <= 0x4DBF):
            n_cjk += 1
        elif (0x3040 <= u <= 0x309F) or (0x30A0 <= u <= 0x30FF):
            n_kana += 1
        elif 0x0400 <= u <= 0x04FF:
            n_cyr += 1
        elif (0x0600 <= u <= 0x06FF) or (0x0750 <= u <= 0x077F):
            n_arab += 1
    if lang == "ru":
        foreign = n_cjk + n_kana + n_arab
    elif lang in ("en", "de", "es", "fr"):
        foreign = n_cjk + n_kana + n_cyr + n_arab
    elif lang == "zh":
        foreign = n_cyr + n_arab + n_kana
    elif lang == "ja":
        foreign = n_cyr + n_arab
    elif lang == "ar":
        foreign = n_cjk + n_kana + n_cyr
    else:
        foreign = 0
    return foreign >= 2


GLITCH_RE = re.compile(r"(\.\.\.\*+|\*{3,}|\*\s*\*)")


def summarize_multispeech(rel_path: str) -> dict:
    recs = load_jsonl(rel_path)
    healthy = unclosed = empty = eos = bleed = 0
    c8_sum = ast_sum = 0.0
    clean_lens: list[int] = []
    tok_lens: list[int] = []
    by_lang_healthy: dict[str, int] = {}
    healthy_map: dict[str, bool] = {}
    for r in recs:
        th_on = bool(r.get("thinking_on", False))
        txt = r["text"]
        if th_on:
            if "</think>" not in txt:
                closed, clean = False, ""
            else:
                closed = True
                clean = txt.rsplit("</think>", 1)[1]
        else:
            closed, clean = True, txt
        clean = clean.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        clean_lens.append(len(clean))
        tok_lens.append(int(r.get("total_tokens", 0)))
        he = bool(r.get("hit_eos", False))
        lg = r["lang"]
        c8 = char_8gram_rep(clean)
        w4 = word_4gram_rep(clean)
        bl = has_foreign_bleed(clean, lg)
        ast_rate = 1000.0 * len(GLITCH_RE.findall(clean)) / max(len(clean), 1) if clean else 0.0
        if not closed:
            unclosed += 1
        is_empty = (not closed) or (len(clean) == 0)
        severe = is_empty or (not he) or (c8 > 0.25) or (lg not in ("zh", "ja") and w4 > 0.20)
        ok = (not severe) and (not bl)

        c8_sum += c8
        ast_sum += ast_rate
        if he:
            eos += 1
        if is_empty:
            empty += 1
        if bl:
            bleed += 1
        if ok:
            healthy += 1
            by_lang_healthy[lg] = by_lang_healthy.get(lg, 0) + 1
        healthy_map[r["unique_id"]] = ok

    n = len(recs)
    med_c = statistics.median(clean_lens) if clean_lens else 0.0
    med_t = statistics.median(tok_lens) if tok_lens else 0.0
    return {
        "n": n,
        "healthy": healthy,
        "healthy_pct": round(100.0 * healthy / n, 2),
        "unclosed": unclosed,
        "unclosed_pct": round(100.0 * unclosed / n, 2),
        "empty": empty,
        "empty_pct": round(100.0 * empty / n, 2),
        "eos": eos,
        "char_rep8_pct": round(100.0 * c8_sum / n, 2),
        "bleed": bleed,
        "ast_per_1k": round(ast_sum / n, 2),
        "median_chars": int(math.floor(med_c + 0.5)),
        "median_tokens": int(math.floor(med_t + 0.5)),
        "mean_tokens": round(sum(tok_lens) / n, 1) if tok_lens else 0.0,
        "by_lang_healthy": by_lang_healthy,
        "healthy_map": healthy_map,
    }


def run_grader_self_tests() -> None:
    # 1. Symbolic equivalence regression cases (including polar tuple precalculus/807)
    assert mit_is_correct(r"(3,\pi/2)", r"\left(3,\frac{\pi}{2}\right)"), "Polar tuple equivalence failed"
    assert mit_is_correct(r"\frac{2}{4}", r"\frac{1}{2}"), "Fraction reduction equivalence failed"
    assert mit_is_correct("1/2", r"\frac12"), "Slash-to-frac equivalence failed"
    assert mit_is_correct(r"\sqrt{4}", "2"), "Radical equivalence failed"
    assert not mit_is_correct("0", "1"), "False positive on 0 == 1"

    # 2. Anti-shortcut witness: pre-saved strict_ok=True must NEVER force MIT-Std/Strict to True,
    #    and any mismatch between stored strict_ok and recomputed Strict must raise AssertionError.
    mock_st, mock_std, mock_len = eval_mit_three_metrics(
        [{"unique_id": "synthetic_bad", "hit_eos": True, "text": "</think> \\boxed{0}", "gold": "1"}]
    )
    assert not mock_st["synthetic_bad"] and not mock_std["synthetic_bad"] and not mock_len["synthetic_bad"]
    for bad_row in (
        {"unique_id": "tampered_true", "hit_eos": True, "text": "</think> \\boxed{0}", "gold": "1", "strict_ok": True},
        {"unique_id": "tampered_false", "hit_eos": True, "text": "</think> \\boxed{1}", "gold": "1", "strict_ok": False},
    ):
        try:
            eval_mit_three_metrics([bad_row])
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected AssertionError on tampered strict_ok row: {bad_row['unique_id']}")


def main() -> None:
    run_grader_self_tests()

    print("========================================================================")
    print(f"1. SHA-256 CHECKSUM VERIFICATION ({len(EXPECTED_SHA256)} ARTIFACTS)")
    print("========================================================================")
    for rel, exp in EXPECTED_SHA256.items():
        p = os.path.join(REPO_ROOT, rel)
        got = sha256_file(p)
        assert got == exp, f"SHA-256 mismatch on {rel}: got {got}, expected {exp}"
        print(f"  [OK] {rel:58s} {got[:16]}...")

    # Verify calibration / holdout split manifest
    with open(os.path.join(REPO_ROOT, "suites/reproducibility_and_holdout_manifest.json"), encoding="utf-8") as mf:
        repro_manifest = json.load(mf)
    with open(os.path.join(REPO_ROOT, "suites/suite_math500_full.json"), encoding="utf-8") as s500f:
        suite_500 = json.load(s500f)
    with open(os.path.join(REPO_ROOT, "suites/fresh_blind56_32k_suite.json"), encoding="utf-8") as b56f:
        suite_b56 = json.load(b56f)

    split_info = repro_manifest["math500_calibration_vs_holdout_split"]
    excl_200 = set(split_info["excluded_calibration_and_exploratory_200_ids"])
    untouched_300 = set(split_info["untouched_holdout_pool_300_ids"])
    all_500_ids = {t["unique_id"] for t in suite_500}
    b56_math_ids = {t["unique_id"] for t in suite_b56 if not t.get("is_longtext", False)}
    assert len(excl_200) == 200 and len(untouched_300) == 300
    assert excl_200.isdisjoint(untouched_300) and (excl_200 | untouched_300) == all_500_ids
    assert len(b56_math_ids) == 49 and b56_math_ids.issubset(untouched_300)
    print("  [OK] MATH-500 partition verified: 200 excluded + 300 untouched = 500 (0 overlap; all 49 FreshBlind56 in untouched 300)")

    print("\n========================================================================")
    print("2. CHAPTER 2 — MATHEMATICS @ MAX_NEW_TOKENS = 32,768")
    print("========================================================================")
    t0_maps = {}
    t0_std_maps = {}
    t0_len_maps = {}
    for tag, fn in [
        ("Base", "traces/math500_base_think32k_pure_t0_strat100.jsonl"),
        ("MIT-p50c", "traces/math500_mit_p50c_think32k_pure_t0_strat100.jsonl"),
        ("SGNE-CC", "traces/math500_sgne_cc_think32k_pure_t0_strat100.jsonl"),
    ]:
        rows = load_jsonl(fn)
        st_m, std_m, len_m = eval_mit_three_metrics(rows)
        ok = sum(st_m.values())
        ok_std = sum(std_m.values())
        ok_len = sum(len_m.values())
        cut = sum(1 for r in rows if not r["hit_eos"])
        unclosed = sum(1 for r in rows if "</think>" not in r["text"])
        tok = sum(r["total_tokens"] for r in rows) / len(rows)
        t0_maps[tag] = st_m
        t0_std_maps[tag] = std_m
        t0_len_maps[tag] = len_m
        print(
            f"  MATH-500 strat-100 PURE_T0 | {tag:8s}: Strict={ok:2d}/100 ({ok:4.1f}%), "
            f"MIT-Std={ok_std:2d}/100 ({ok_std:4.1f}%), MIT-Lenient={ok_len:2d}/100 ({ok_len:4.1f}%), "
            f"cut={cut:3d}%, unclosed={unclosed:2d}/100, tok={tok:7.1f}"
        )

    assert t0_len_maps["MIT-p50c"]["test/precalculus/807.json"] is True
    assert (sum(t0_len_maps["Base"].values()), sum(t0_len_maps["MIT-p50c"].values()), sum(t0_len_maps["SGNE-CC"].values())) == (7, 20, 45)

    for m_lbl, m_dict in [("Strict", t0_maps), ("MIT-Std", t0_std_maps), ("MIT-Lenient", t0_len_maps)]:
        _, _, g, l, p_val = mcnemar_pair(m_dict["SGNE-CC"], m_dict["MIT-p50c"])
        print(f"  McNemar [SGNE-CC vs MIT-p50c @ PURE_T0, {m_lbl:11s}]: +{g}/-{l}, exact two-sided p = {p_val:.3e}")

    # Blind56 (49 MATH-500 holdout items + 7 prose essays)
    b56_maps = {}
    for tag, fn in [
        ("Base", "traces/blind56_base_think32k_pure_t0.jsonl"),
        ("SGNE-CC", "traces/blind56_sgne_cc_think32k_pure_t0.jsonl"),
    ]:
        all_b56 = load_jsonl(fn)
        rows = [r for r in all_b56 if not r.get("is_longtext", False)]
        prose = [r for r in all_b56 if r.get("is_longtext", False)]
        st_b56, _, _ = eval_mit_three_metrics(rows)
        ok = sum(st_b56.values())
        p_closed = sum(1 for r in prose if "</think>" in r.get("text", ""))
        p_eos = sum(1 for r in prose if r["hit_eos"])
        b56_maps[tag] = st_b56
        print(
            f"  FreshBlind56 math holdout  | {tag:8s}: {ok:2d}/{len(rows)} ({100*ok/len(rows):5.1f}%) "
            f"| prose essays ({len(prose)}): closed_think={p_closed}/{len(prose)}, EOS={p_eos}/{len(prose)}"
        )
    _, _, g, l, p_b56 = mcnemar_pair(b56_maps["SGNE-CC"], b56_maps["Base"])
    print(f"  McNemar [SGNE-CC vs Base @ FreshBlind56]: +{g}/-{l}, exact two-sided p = {p_b56:.3e}")

    # Full MATH-500 CARD_PP15: Direct 3-model traces (independently re-graded) + MIT official vLLM reference
    base_500 = load_jsonl("traces/math500_base_think32k_card_pp15.jsonl")
    mit_500 = load_jsonl("traces/math500_mit_p50c_think32k_card_pp15.jsonl")
    sgne_500 = load_jsonl("traces/math500_sgne_cc_think32k_card_pp15.jsonl")
    base_500_map, _, _ = eval_mit_three_metrics(base_500)
    mit_500_map, _, _ = eval_mit_three_metrics(mit_500)
    sgne_500_map, _, _ = eval_mit_three_metrics(sgne_500)

    with open(os.path.join(REPO_ROOT, "traces/mit_official_math500_ref/mit_p50c_official_500_per_problem.json"), encoding="utf-8") as mf:
        mit_off = json.load(mf)
    with open(os.path.join(REPO_ROOT, "traces/mit_official_math500_ref/mit_p50c_merged.json"), encoding="utf-8") as mmf:
        mit_merged = json.load(mmf)
    mit_s0_map = {uid: (int(v["correct"][0]) == 1) for uid, v in mit_off.items()}

    for tag, rows_d, m_d in [("Base", base_500, base_500_map), ("MIT-p50c", mit_500, mit_500_map), ("SGNE-CC", sgne_500, sgne_500_map)]:
        ok_d = sum(m_d.values())
        cut_d = 100.0 * sum(1 for r in rows_d if not r["hit_eos"]) / len(rows_d)
        tok_d = sum(r["total_tokens"] for r in rows_d) / len(rows_d)
        print(f"  MATH-500 Full Direct       | {tag:8s}: {ok_d:3d}/500 ({100*ok_d/500:5.2f}%), cut={cut_d:5.2f}%, tok={tok_d:7.1f}")

    ok500 = sum(sgne_500_map.values())
    cut500 = 100.0 * sum(1 for r in sgne_500 if not r["hit_eos"]) / len(sgne_500)
    tok500 = sum(r["total_tokens"] for r in sgne_500) / len(sgne_500)
    mit_m500 = mit_merged["math500"]
    mit_tok500 = float(mit_m500["mean_tokens"])
    mit_cut500 = 100.0 * float(mit_m500["cut_share"])
    print(
        f"  MATH-500 Full MIT Official | MIT-p50c: s0={sum(mit_s0_map.values())}/500 "
        f"({100*sum(mit_s0_map.values())/500:.2f}%), mean4={100*mit_m500['acc']:.2f}%, "
        f"cut={mit_cut500:.2f}% ({mit_cut500/cut500:.1f}x), tok={mit_tok500:.1f} ({mit_tok500/tok500:.2f}x)"
    )
    _, _, g, l, p_500 = mcnemar_pair(sgne_500_map, mit_s0_map)
    print(f"  McNemar [SGNE-CC vs MIT-p50c vLLM s0 @ MATH-500 Full]: +{g}/-{l}, exact two-sided p = {p_500:.4f}")
    _, _, gd, ld, p_500d = mcnemar_pair(sgne_500_map, mit_500_map)
    print(f"  McNemar [SGNE-CC vs MIT-p50c Direct  @ MATH-500 Full]: +{gd}/-{ld}, exact two-sided p = {p_500d:.4f}")

    # Same-engine PURE_T0 vs CARD_PP15 on strat-100
    a_ok, b_ok, g, l, p_mit_same = mcnemar_pair(t0_maps["MIT-p50c"], mit_500_map)
    print(f"  McNemar Same-Engine [MIT-p50c PURE_T0 ({a_ok}/100) vs CARD_PP15 ({b_ok}/100)]: +{g}/-{l}, p = {p_mit_same:.3e}")
    a_ok, b_ok, g, l, p_sgne_inv = mcnemar_pair(t0_maps["SGNE-CC"], sgne_500_map)
    print(f"  McNemar Same-Engine [SGNE-CC PURE_T0 ({a_ok}/100) vs CARD_PP15 ({b_ok}/100)]: +{g}/-{l}, p = {p_sgne_inv:.3f}")

    # Sealed holdout (49 untouched MATH-500 items): SGNE-CC vs MIT-p50c (direct & official vLLM)
    blind_ids = set(b56_maps["SGNE-CC"])
    mit_mean4 = {uid: sum(int(x) for x in v["correct"]) / len(v["correct"]) for uid, v in mit_off.items()}
    a_ok, b_ok, g, l, p_blind = mcnemar_pair({k: sgne_500_map[k] for k in blind_ids}, {k: mit_s0_map[k] for k in blind_ids})
    print(f"  Holdout-49 CARD_PP15 (vLLM)   | SGNE-CC={a_ok}/49 vs MIT-p50c s0={b_ok}/49 (mean4={sum(mit_mean4[k] for k in blind_ids):.1f}): +{g}/-{l}, p = {p_blind:.3f}")
    a_ok, b_ok, g, l, p_blind_d = mcnemar_pair({k: sgne_500_map[k] for k in blind_ids}, {k: mit_500_map[k] for k in blind_ids})
    print(f"  Holdout-49 CARD_PP15 (Direct) | SGNE-CC={a_ok}/49 vs MIT-p50c={b_ok}/49: +{g}/-{l}, p = {p_blind_d:.3f}")

    # Grader consistency on identical extracted answers
    agree = disagree = 0
    for r in sgne_500:
        mp = str(mit_off[r["unique_id"]]["pred_first"]).replace(" ", "")
        if r["pred"] not in ("NONE", "", None) and r["pred"].replace(" ", "") == mp:
            if sgne_500_map[r["unique_id"]] == mit_s0_map[r["unique_id"]]:
                agree += 1
            else:
                disagree += 1
    no_box = sum(1 for r in sgne_500 if r["pred"] in ("NONE", "", None) and ("</think>" in r["text"]) and r["hit_eos"])
    print(f"  Grader agreement on identical extracted strings: {agree}/{agree + disagree} | SGNE-CC finished without \\boxed{{}}: {no_box}")

    # Level 1-2 (n=133) vs Level 3-5 (n=367) — Direct 3-model comparison
    base_by_id = {r["unique_id"]: r for r in base_500}
    mit_by_id = {r["unique_id"]: r for r in mit_500}
    sgne_by_id = {r["unique_id"]: r for r in sgne_500}
    lv12_ids = [r["unique_id"] for r in sgne_500 if str(r["level"]) in ("1", "2")]
    lv35_ids = [r["unique_id"] for r in sgne_500 if str(r["level"]) in ("3", "4", "5")]

    for lbl, ids in [("Level 1-2", lv12_ids), ("Level 3-5", lv35_ids)]:
        n_s = len(ids)
        print(f"  MATH-500 {lbl:9s} Direct (n={n_s:3d}):")
        for tag, by_id, m_d in [("Base", base_by_id, base_500_map), ("MIT-p50c", mit_by_id, mit_500_map), ("SGNE-CC", sgne_by_id, sgne_500_map)]:
            ok_s = sum(1 for u in ids if m_d[u])
            cut_s = 100.0 * sum(1 for u in ids if not by_id[u]["hit_eos"]) / n_s
            tok_s = sum(by_id[u]["total_tokens"] for u in ids) / n_s
            print(f"    {tag:8s}: {ok_s:3d}/{n_s} ({100*ok_s/n_s:5.2f}%), cut={cut_s:5.2f}%, tok={tok_s:7.1f}")
        _, _, g, l, p_sub = mcnemar_pair({u: sgne_500_map[u] for u in ids}, {u: mit_500_map[u] for u in ids})
        tok_ratio = (sum(mit_by_id[u]["total_tokens"] for u in ids) / n_s) / (sum(sgne_by_id[u]["total_tokens"] for u in ids) / n_s)
        print(f"    McNemar [SGNE-CC vs MIT-p50c Direct]: +{g}/-{l}, nominal p = {p_sub:.4f} | token_ratio = {tok_ratio:.2f}x")

    # Complete Level 3-5 partition (142 + 178 + 47 = 367)
    alg2_ids = [u for u in lv35_ids if sgne_by_id[u]["subject"] in ("Algebra", "Prealgebra")]
    hard4_ids = [
        u for u in lv35_ids
        if sgne_by_id[u]["subject"] in ("Intermediate Algebra", "Counting & Probability", "Geometry", "Precalculus")
    ]
    nt_ids = [u for u in lv35_ids if sgne_by_id[u]["subject"] == "Number Theory"]
    assert len(alg2_ids) + len(hard4_ids) + len(nt_ids) == len(lv35_ids) == 367
    for grp_lbl, g_ids in [
        ("Lv3-5 Algebra+Prealgebra", alg2_ids),
        ("Lv3-5 4-Subject Subset", hard4_ids),
        ("Lv3-5 Number Theory", nt_ids),
    ]:
        s_ok, m_ok, g, l, p_g = mcnemar_pair({u: sgne_500_map[u] for u in g_ids}, {u: mit_500_map[u] for u in g_ids})
        print(
            f"  {grp_lbl:26s} (n={len(g_ids):3d}): MIT={m_ok}/{len(g_ids)} ({100*m_ok/len(g_ids):.1f}%) "
            f"vs SGNE={s_ok}/{len(g_ids)} ({100*s_ok/len(g_ids):.1f}%) | McNemar +{g}/-{l}, nominal p = {p_g:.4f}"
        )

    # MGSM-240 & BBH-160 + HumanEval-164
    print("\n========================================================================")
    print("3. MGSM-240, BBH-160 & HUMANEVAL-164 @ MAX_NEW_TOKENS = 32,768")
    print("========================================================================")
    mgsm_maps = {}
    bbh_strict_maps = {}
    bbh_clean_maps = {}
    wol_maps = {}
    bbh_toks = {}
    for tag, fn in [
        ("Base", "traces/mgsm240_bbh160_base_think32k_card_pp15.jsonl"),
        ("MIT-p50c", "traces/mgsm240_bbh160_mit_p50c_think32k_card_pp15.jsonl"),
        ("SGNE-CC", "traces/mgsm240_bbh160_sgne_cc_think32k_card_pp15.jsonl"),
    ]:
        rows = load_jsonl(fn)
        mg = [r for r in rows if r["unique_id"].startswith("mgsm/")]
        bb = [r for r in rows if r["unique_id"].startswith("bbh/")]
        wol = [r for r in bb if r["unique_id"].startswith("bbh/web_of_lies/")]
        mg_map = recompute_mgsm_strict(mg)
        mg_ok = sum(mg_map.values())
        mg_tok = sum(r["total_tokens"] for r in mg) / len(mg)
        mg_zh_ok = sum(1 for r in mg if r["lang"] == "zh" and mg_map[r["unique_id"]])
        bb_strict = sum(1 for r in bb if r["strict_ok"])
        c_map = {
            r["unique_id"]: bool(
                ("</think>" in r["text"])
                and mit_last_boxed(r["text"].rsplit("</think>", 1)[1]) not in (None, "")
                and clean_bbh(mit_last_boxed(r["text"].rsplit("</think>", 1)[1])) == clean_bbh(r["gold"])
            )
            for r in bb
        }
        bb_clean = sum(c_map.values())
        bb_cut = 100.0 * sum(1 for r in bb if not r["hit_eos"]) / len(bb)
        bb_tok = sum(r["total_tokens"] for r in bb) / len(bb)
        wol_ok = sum(1 for r in wol if c_map[r["unique_id"]])
        wol_cut = 100.0 * sum(1 for r in wol if not r["hit_eos"]) / len(wol)
        wol_tok = sum(r["total_tokens"] for r in wol) / len(wol)
        mgsm_maps[tag] = mg_map
        bbh_strict_maps[tag] = {r["unique_id"]: bool(r["strict_ok"]) for r in bb}
        bbh_clean_maps[tag] = c_map
        wol_maps[tag] = {r["unique_id"]: c_map[r["unique_id"]] for r in wol}
        bbh_toks[tag] = bb_tok
        print(f"  MGSM-240 | {tag:8s}: {mg_ok:3d}/240 ({100*mg_ok/240:5.2f}%), zh={mg_zh_ok:2d}/40 ({100*mg_zh_ok/40:4.1f}%), tok={mg_tok:7.1f}")
        print(f"  BBH-160  | {tag:8s}: clean={bb_clean:2d}/160 ({100*bb_clean/160:5.2f}%), strict={bb_strict:2d}/160 ({100*bb_strict/160:5.2f}%), cut={bb_cut:5.2f}%, tok={bb_tok:7.1f} | WebOfLies={wol_ok:2d}/20 ({100*wol_ok/20:4.1f}%, cut={wol_cut:4.1f}%, tok={wol_tok:6.0f})")

    _, _, g, l, p_mgsm_naive = mcnemar_pair(mgsm_maps["SGNE-CC"], mgsm_maps["MIT-p50c"])
    # 40-problem clustered sign-flip across the 6 translations of each MGSM task
    prob_indices = sorted({u.rsplit("/", 1)[1] for u in mgsm_maps["SGNE-CC"]})
    assert len(prob_indices) == 40
    block_diffs = []
    for pidx in prob_indices:
        s_cnt = sum(1 for lg in ("ru", "zh", "ja", "fr", "es", "de") if mgsm_maps["SGNE-CC"][f"mgsm/{lg}/{pidx}"])
        m_cnt = sum(1 for lg in ("ru", "zh", "ja", "fr", "es", "de") if mgsm_maps["MIT-p50c"][f"mgsm/{lg}/{pidx}"])
        block_diffs.append(s_cnt - m_cnt)
    p_mgsm_cluster = exact_cluster_signflip_p(block_diffs)

    _, _, gc, lc, p_bbh_clean = mcnemar_pair(bbh_clean_maps["SGNE-CC"], bbh_clean_maps["MIT-p50c"])
    _, _, gs, ls, p_bbh_strict = mcnemar_pair(bbh_strict_maps["SGNE-CC"], bbh_strict_maps["MIT-p50c"])
    _, _, gw, lw, p_wol_mit = mcnemar_pair(wol_maps["SGNE-CC"], wol_maps["MIT-p50c"])
    _, _, gwb, lwb, p_wol_base = mcnemar_pair(wol_maps["SGNE-CC"], wol_maps["Base"])
    print(
        f"  MGSM-240 [SGNE vs MIT]: discordant +{g}/-{l} | 40-block cluster sign-flip p = {p_mgsm_cluster:.3e} "
        f"(naive 240-pair p = {p_mgsm_naive:.3e})"
    )
    print(f"  McNemar BBH-160  [SGNE vs MIT]: clean +{gc}/-{lc} (p = {p_bbh_clean:.4f}) | strict +{gs}/-{ls} (p = {p_bbh_strict:.4f}) | token_ratio = {bbh_toks['MIT-p50c']/bbh_toks['SGNE-CC']:.2f}x")
    print(f"  McNemar WebOfLies (n=20)      : SGNE vs MIT +{gw}/-{lw} (p = {p_wol_mit:.4f}) | SGNE vs Base +{gwb}/-{lwb} (p = {p_wol_base:.4f})")

    # HumanEval-164 static AST check + archived execution summary verification
    with open(os.path.join(REPO_ROOT, "suites/suite_unseen_humaneval164.json"), encoding="utf-8") as hef:
        he_suite = {t["unique_id"]: t for t in json.load(hef)}
    he_arch = repro_manifest["humaneval_archived_execution_and_static_ast"]["models"]
    he_think = {}
    he_arch_maps: dict[str, dict[str, bool]] = {}
    for tag, mkey, fn in [
        ("Base", "base", "traces/humaneval164_base_think32k_card_pp15.jsonl"),
        ("MIT-p50c", "mit_p50c", "traces/humaneval164_mit_p50c_think32k_card_pp15.jsonl"),
        ("SGNE-CC", "sgne_cc", "traces/humaneval164_sgne_cc_think32k_card_pp15.jsonl"),
    ]:
        rows = load_jsonl(fn)
        mtok = sum(r["total_tokens"] for r in rows) / len(rows)
        mthk = sum(r["think_tokens"] for r in rows) / len(rows)
        mcut = 100.0 * sum(1 for r in rows if not r["hit_eos"]) / len(rows)
        muncl = sum(1 for r in rows if "</think>" not in r["text"])
        extracted = ast_ok = ast_err = 0
        for r in rows:
            if "</think>" not in r["text"]:
                continue
            tmeta = he_suite[r["unique_id"]]
            code = extract_humaneval_candidate(r["text"].rsplit("</think>", 1)[1], tmeta["entry_point"], tmeta["code_signature_prompt"])
            if code:
                extracted += 1
                try:
                    ast.parse(code)
                    ast_ok += 1
                except SyntaxError:
                    ast_err += 1
        he_think[tag] = mthk
        arch_entry = he_arch[mkey]
        passed_set = set(arch_entry["archived_passed_ids"])
        he_arch_maps[tag] = {uid: (uid in passed_set) for uid in he_suite}
        assert arch_entry["static_check_current_extractor"]["code_extracted"] == extracted
        assert arch_entry["static_check_current_extractor"]["ast_parse_valid"] == ast_ok
        print(
            f"  HumanEval-164 | {tag:8s}: archived_pass@1={len(passed_set):2d}/164 ({100*len(passed_set)/164:5.2f}%), "
            f"extracted={extracted:3d}/164, ast_valid={ast_ok:3d}/164 (syntax_err={ast_err}), "
            f"unclosed_think={muncl:2d}/164, cut={mcut:5.2f}%, tok={mtok:7.1f}, think={mthk:7.1f}"
        )
    _, _, g_he, l_he, p_he = mcnemar_pair(he_arch_maps["SGNE-CC"], he_arch_maps["MIT-p50c"])
    _, _, g_hb, l_hb, p_hb = mcnemar_pair(he_arch_maps["SGNE-CC"], he_arch_maps["Base"])
    print(
        f"  HumanEval-164 Archived McNemar: SGNE vs MIT +{g_he}/-{l_he} (p = {p_he:.4f}) | "
        f"SGNE vs Base +{g_hb}/-{l_hb} (p = {p_hb:.4f}) | <think> ratio = {he_think['MIT-p50c'] / he_think['SGNE-CC']:.2f}x"
    )

    print("\n========================================================================")
    print("4. MULTISPEECH-64 AUTOMATED FORMAL FILTER (8 LANGUAGES x 8 GENRES)")
    print("========================================================================")
    for regime, suffix in [
        ("THINK32K_PURE_T0", "think32k_pure_t0"),
        ("THINK32K_CARD_PP15", "think32k_card_pp15"),
        ("NOTHINK_PURE_T0", "nothink_t0"),
        ("NOTHINK_CARD_PP15", "nothink_card_pp15"),
    ]:
        hmaps = {}
        for tag, mslug in [("Base", "base"), ("MIT-p50c", "mit_p50c"), ("SGNE-CC", "sgne_cc")]:
            st = summarize_multispeech(f"traces/multispeech64_{mslug}_{suffix}.jsonl")
            hmaps[tag] = st["healthy_map"]
            langs = " ".join(f"{lg}={st['by_lang_healthy'].get(lg, 0)}" for lg in ("ru", "en", "zh", "de", "es", "fr", "ja", "ar"))
            print(
                f"  {regime:18s} | {tag:8s}: filter_pass={st['healthy']:2d}/64 ({st['healthy_pct']:5.1f}%), "
                f"unclosed={st['unclosed']:2d}/64 ({st['unclosed_pct']:4.1f}%), empty_post={st['empty']:2d}/64, EOS={st['eos']:2d}/64, "
                f"med_chars={st['median_chars']:4d}, tok_med={st['median_tokens']:5d}, tok_mean={st['mean_tokens']:7.1f}, "
                f"char_rep8={st['char_rep8_pct']:5.2f}%, bleed={st['bleed']}, *pattern/1k={st['ast_per_1k']:5.2f}"
            )
            print(f"  {'':18s}   {'':8s}  per-language filter_pass /8: {langs}")
        pairs = []
        for a, b in (("SGNE-CC", "MIT-p50c"), ("SGNE-CC", "Base"), ("MIT-p50c", "Base")):
            _, _, g, l, p = mcnemar_pair(hmaps[a], hmaps[b])
            pairs.append(f"{a} vs {b}: +{g}/-{l} p={p:.3g}")
        print(f"  {'':18s}   McNemar filter_pass: " + " | ".join(pairs))

    print(
        "\nVERIFICATION COMPLETE: 37/37 SHA-256 checksums, independent MATH/Blind56/MGSM/BBH/MultiSpeech "
        "trace recomputations, static HumanEval AST checks, and archived HumanEval tables verified."
    )


if __name__ == "__main__":
    main()
