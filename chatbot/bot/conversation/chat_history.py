from __future__ import annotations
from typing import List, Optional, Callable, Tuple, Dict
from dataclasses import dataclass
import os, re, logging

# Chat buffer (single responsibility)
class ChatHistory(list):
    """
    A tiny sliding-window chat buffer.
    - total_length > 0: keep only the last N messages (drop oldest first)
    - total_length <= 0: unlimited (no trimming)
    """
    def __init__(self, messages: Optional[List[str]] = None, total_length: int = 2):
        super().__init__(messages or [])
        self.total_length = total_length
    
    # Trim when current length >= window so we never exceed the cap.
    def append(self, msg: str):
        if self.total_length and self.total_length > 0:
            while len(self) >= self.total_length:
                self.pop(0)
        super().append(msg)

    # Return the last n messages as a single string.
    def snapshot(self, n: Optional[int] = None) -> str:
        if n is None:
            n = self.total_length if (self.total_length and self.total_length > 0) else len(self)
        n = max(1, min(n, len(self))) if self else 0
        return "\n".join(self[-n:]) if n else ""

    # Return the last k messages as a list (most recent last).
    def recent(self, k: int = 3) -> List[str]:  
        if k <= 0 or not self:
            return []
        k = min(k, len(self))
        return list(self[-k:])
    # Safer default: show only the last window to avoid leaking stale prompts.
    def __str__(self) -> str:
        return self.snapshot()

# Thresholds & constants (env-overridable)
TOPIC_K              = int(os.getenv("TOPIC_K", "3"))          # how many recent user msgs to consider
COS_THRES_SOFT       = float(os.getenv("COS_THRES_SOFT", "0.60"))
SCORE_THRES_DECIDE   = float(os.getenv("SCORE_THRES_DECIDE", "0.70"))
JACCARD_FALLBACK_MIN = float(os.getenv("JACCARD_FALLBACK_MIN", "0.35"))

_STOP = { # Basic stopwords for token/Jaccard fallback
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are",
    "what","why","how","which","that","this","it","as","at","from","about"
}

_FIN_KWS = { # Financial keywords for entity overlap
    "revenue","eps","earnings","guidance","margin","gross","operating","cash","capex",
    "10-k","10-q","8-k","q1","q2","q3","q4","fy","yoy","qoq","gaap","non-gaap"
}
_YR_RE  = re.compile(r"\b(20\d{2}|19\d{2})\b")
_QTR_RE = re.compile(r"\b(q[1-4])\b", re.I)
_TKR_RE = re.compile(r"\b[A-Z]{1,5}\b")  # loose ticker-like token

# Context carrier to avoid long parameter lists
@dataclass
# Aggregates all inputs required to decide if chat history should be used.
class HistoryContext: 
    curr_text: str # Current user query text.
    curr_ticker: Optional[str] # Current target ticker symbol (if any).
    intent: dict # Parsed intent dict (e.g., {"price":bool,"metric":bool,"explain":bool,"compare":bool}).
    last_ticker: Optional[str] = None # Ticker symbol from the previous turn, used to detect topic switches.
    recent_user_messages: Optional[List[str]] = None  # A list of recent user messages. Either most-recent-first or last is acceptable.
    force_new_topic: bool = False # If True, force history exclusion for this turn.
    force_followup: bool = False # If True, slightly relax thresholds for this turn.
    embedder: Optional[Callable[[List[str]], List[List[float]]]] = None # Optional embedding function that maps a list[str] -> list[vector].
    extra_fin_kws: Optional[List[str]] = None # Optional extra domain keywords to include in entity matching.

# Public API — Primary gating

#  Risk-minimized gate to decide whether to include chat history for the current turn.
def should_use_history(ctx: HistoryContext, trace: bool = False, logger: Optional[logging.Logger] = None) -> bool:
    log = logger.info if (trace and logger) else (print if trace else (lambda *args, **kw: None))
    # (1) Hard rules
    if not _hard_rules_pass(ctx):
        log("[HISTORY] skip: hard rules failed")
        return False

    # (2) Collect candidates (latest first), keep only last-K distinct from current text
    cands = _collect_candidates(ctx.recent_user_messages or [], ctx.curr_text, TOPIC_K)
    if not cands:
        log("[HISTORY] skip: no candidates")
        return False

    # (3) Scoring signals
    fin_kws = _build_fin_kws(ctx.extra_fin_kws)
    best = _score_best_candidate(ctx.curr_text, cands, fin_kws, ctx.embedder)

    # (4) Thresholds and final decision
    thr    = _adaptive_score_threshold(ctx.intent, ctx.force_followup)
    cos_ok = (best["cos"] >= COS_THRES_SOFT) if callable(ctx.embedder) else False
    jac_ok = (best["jac"] >= JACCARD_FALLBACK_MIN)
    ent_ok = (best["ent"] >= 1)

    decide = (best["score"] >= thr) and ((cos_ok or jac_ok) or ctx.force_followup) and (ent_ok or ctx.force_followup)

    if trace:
        log("[HISTORY] best_idx=%s score=%.3f cos=%.3f jac=%.3f ent=%s rec=%.2f thr=%.2f",
            str(best["idx"]), best["score"], best["cos"], best["jac"], best["ent"], best["rec"], thr)
        log(f"[HISTORY] {'use_history=True ← ' + best['text'][:160] if decide else 'use_history=False'}")
    return bool(decide)

# Public API — Optional 3-level gating

# Decide the level of history usage for the current turn.
def decide_history_level(ctx: HistoryContext) -> Tuple[str, Dict[str, float]]: 
    """
    Return ("none" | "selective" | "full", meta).
      - "full"      : pass full history window
      - "selective" : pass top-1 prior user message
      - "none"      : pass nothing
    """
    if not _hard_rules_pass(ctx):
        return "none", {"reason": "hard_rules_failed"}

    cands = _collect_candidates(ctx.recent_user_messages or [], ctx.curr_text, TOPIC_K)
    if not cands:
        return "none", {"reason": "no_candidates"}

    fin_kws = _build_fin_kws(ctx.extra_fin_kws)
    best = _score_best_candidate(ctx.curr_text, cands, fin_kws, ctx.embedder)

    thr    = _adaptive_score_threshold(ctx.intent, ctx.force_followup)
    cos_ok = (best["cos"] >= COS_THRES_SOFT) if callable(ctx.embedder) else False
    jac_ok = (best["jac"] >= JACCARD_FALLBACK_MIN)
    ent_ok = (best["ent"] >= 1)

    score_ok = (best["score"] >= thr)
    sim_ok   = (cos_ok or jac_ok)
    ent_gate = ent_ok

    if (score_ok and (sim_ok or ctx.force_followup) and (ent_gate or ctx.force_followup)):
        return "full", best

    lower = max(0.0, thr - 0.10)  # A narrow “almost there” middle band for selective history
    if (best["score"] >= lower) and (sim_ok or ctx.force_followup) and (ent_gate or ctx.force_followup):
        return "selective", best

    return "none", best

# Public API — Optional inspection utility
# Compute signals between two texts for debugging/inspection.
def compute_similarity_signals(curr_text: str, cand_text: str,
                               embedder: Optional[Callable[[List[str]], List[List[float]]]] = None) -> Dict[str, float]:
    fin_kws = _build_fin_kws(None) # Build financial keywords used for entity overlap.
    cos = _cosine_signal(curr_text, cand_text, embedder) # Cosine similarity using the provided embedder (0.0 if unavailable).
    jac = _jaccard(_tokenize(curr_text), _tokenize(cand_text)) # Jaccard similarity over token sets.
    ent = _entity_overlap(curr_text, cand_text, fin_kws) # Entity overlap count (years/quarters/fin keywords/ticker-like).
    rec = 1.0   # Recency weight (neutral 1.0 for standalone).
    score = _combine_score(cos, ent, rec) # Combined conservative score used by the gate.
    return {"cos": cos, "jac": jac, "ent": float(ent), "rec": rec, "score": score}


# Private helpers (kept together for readability)
def _hard_rules_pass(ctx: HistoryContext) -> bool:
    # Ticker change → always new topic
    if ctx.curr_ticker and ctx.last_ticker and ctx.curr_ticker.upper() != (ctx.last_ticker or "").upper():
        return False
    # No ticker and non-numeric intent → treat as new concept
    if (not ctx.curr_ticker) and (not (ctx.intent.get("price") or ctx.intent.get("metric"))):
        return False
    # Explicitly forced by caller
    if ctx.force_new_topic:
        return False
    return True

# Return up to k latest user messages that are non-empty and not identical to curr_text.
# Input can be most-recent-first or last; we just iterate as provided and cap at k.
def _collect_candidates(messages: List[str], curr_text: str, k: int) -> List[str]:
    out: List[str] = []
    for m in (messages or []):
        if m and m.strip() and m.strip() != curr_text.strip():
            out.append(m)
        if len(out) >= k:
            break
    return out

# Build the set of financial keywords used for entity overlap.
def _build_fin_kws(extra: Optional[List[str]]) -> set[str]: 
    mon = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}
    base = _FIN_KWS | mon
    return base | ({w.lower() for w in extra} if extra else set())

# Tokenize a string into lowercase alphanumeric/dash tokens, and remove basic stopwords and short tokens (<= 2).
def _tokenize(s: str) -> set[str]: 
    return {t for t in re.findall(r"[a-z0-9\-]+", (s or "").lower()) if t not in _STOP and len(t) > 2}

# Compute Jaccard similarity between two token sets. Returns 0.0 if either set is empty.
def _jaccard(a: set[str], b: set[str]) -> float: 
    if not a or not b: return 0.0
    u = a | b
    return len(a & b) / (len(u) or 1)

# Compute cosine similarity for two numeric vectors without external deps.
def _cosine(a, b) -> float:
    try:
        dot = na = nb = 0.0                  # initialize dot product, norm(a), norm(b)
        for x, y in zip(a, b):               # iterate over elements of both vectors
            fx = float(x); fy = float(y)     # ensure values are floats
            dot += fx * fy                   # accumulate dot product
            na += fx * fx                    # accumulate squared norm of vector a
            nb += fy * fy                    # accumulate squared norm of vector b
        denom = (na ** 0.5) * (nb ** 0.5)    # compute denominator = ||a|| * ||b||
        return float(dot / denom) if denom > 0.0 else 0.0  # return cosine similarity
    except Exception: 
        return 0.0  # Returns 0.0 for any failure or zero-norm vectors.

# Embed two texts using the provided embedder and return cosine similarity.
def _cosine_signal(a_text: str, b_text: str,
                   embedder: Optional[Callable[[List[str]], List[List[float]]]]) -> float:
    if not callable(embedder):              # check if embedder is callable
        return 0.0
    try:
        va, vb = embedder([a_text])[0], embedder([b_text])[0]  # compute embeddings
        return _cosine(va, vb)              # calculate cosine similarity between embeddings
    except Exception:
        return 0.0 # Returns 0.0 if the embedder is unavailable or fails.

# Extract a lightweight set of entities from text
def _entities_with_kws(text: str, fin_kws: set[str]) -> set[str]:
    s_low = (text or "").lower()                          # normalize to lowercase
    ents = set(t.lower() for t in _QTR_RE.findall(text))  # extract quarter entities (q1..q4)
    ents |= set(_YR_RE.findall(text))                     # extract year entities (e.g., 2024)
    ents |= {w for w in fin_kws if w in s_low}            # add financial keywords if present
    ents |= set(t for t in _TKR_RE.findall(text) if t.isupper())  # detect ticker-like tokens
    return ents

#  Return the size of the intersection between two entity sets extracted from a_text and b_text.
def _entity_overlap(a_text: str, b_text: str, fin_kws: set[str]) -> int:
    return len(_entities_with_kws(a_text, fin_kws) & _entities_with_kws(b_text, fin_kws))

# Convert a candidate's recency (0 = most recent) into a weight in (0, 1].
def _recency_weight(idx_from_most_recent: int) -> float:
    return 1.0 / (idx_from_most_recent + 1)

# Conservative weighting used throughout: score = 0.6 * cosine + 0.3 * min(ent,3)/3 + 0.1 * recency
def _combine_score(cos: float, ent: int, rec: float) -> float:
    return 0.6 * cos + 0.3 * min(ent, 3) / 3.0 + 0.1 * rec

# Adapt the decision threshold based on intent and "follow-up" override:
def _adaptive_score_threshold(intent: dict, force_followup: bool) -> float:
    thr = SCORE_THRES_DECIDE
    if intent.get("price") or intent.get("metric"):  thr -= 0.05
    if intent.get("explain") or intent.get("compare"): thr += 0.05
    if force_followup: thr -= 0.08
    return thr

# Score candidates and return the best one with diagnostics.
def _score_best_candidate(curr_text: str, candidates: List[str],
                          fin_kws: set[str],
                          embedder: Optional[Callable[[List[str]], List[List[float]]]]) -> Dict[str, float]:
    best = {"idx": -1, "cos": 0.0, "jac": 0.0, "ent": 0, "rec": 0.0, "score": 0.0, "text": ""}
    for i, cand in enumerate(candidates):
        cos = _cosine_signal(curr_text, cand, embedder)
        jac = _jaccard(_tokenize(curr_text), _tokenize(cand))
        ent = _entity_overlap(curr_text, cand, fin_kws)
        rec = _recency_weight(i)
        sc  = _combine_score(cos, ent, rec)
        if sc > best["score"]:
            best = {"idx": i, "cos": cos, "jac": jac, "ent": ent, "rec": rec, "score": sc, "text": cand}
    return best
