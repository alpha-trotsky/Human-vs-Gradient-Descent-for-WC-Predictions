"""
Event classification: decide how the contracts of an event relate to each other.

Everything downstream (coherence checks, pricing model choice) is only valid for certain
relationships, so ingestion classifies once and records the result on each MarketContract:

  PARTITION             mutually exclusive legs. Selling every leg is safe whenever legs are
                        mutually exclusive; *buying* every leg additionally needs `exhaustive`
                        (some listed leg is guaranteed to win).
  CUMULATIVE_THRESHOLD  "above X" / "below X" ladders (nested events).
  NESTED_BY_DATE        "by <date>" ladders (nested events).
  UNKNOWN               anything we cannot prove -> no structural checks, no structural pricing.

`exhaustive` is only set when it is proven from the data (strike tiling, venue negRisk flags),
never assumed.
"""
import re
from typing import Iterable, List, Optional, Tuple

from copula_execution.types import EventKind, MarketContract

Edges = Tuple[Optional[float], Optional[float]]
_TOL = 1e-6

_MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")


def contract_edges(c: MarketContract) -> Optional[Edges]:
    """Continuous-domain (lo, hi) region a contract pays on, or None if it carries no numeric structure."""
    if c.model_edges is not None:
        return c.model_edges
    if c.floor_strike is not None or c.cap_strike is not None:
        return (c.floor_strike, c.cap_strike)
    return None


def _set_all(contracts: Iterable[MarketContract], kind: EventKind, exhaustive: bool = False) -> None:
    for c in contracts:
        c.event_kind = kind
        c.exhaustive = exhaustive


# ----------------------------------------------------------------------------------------------
# Kalshi
# ----------------------------------------------------------------------------------------------
def _infer_step(betweens: List[MarketContract]) -> Optional[float]:
    """
    Kalshi brackets on quantised prints (GDP/CPI to 0.1) look like [0.1,0.5], [0.6,1.0]: the gap
    between one cap and the next floor is the print resolution. Returns that step, 0.0 when the
    brackets touch, or None if the gaps are irregular (a bracket is probably missing).
    """
    bs = sorted(betweens, key=lambda c: c.floor_strike)
    gaps = [round(b.floor_strike - a.cap_strike, 9) for a, b in zip(bs, bs[1:])]
    pos = [g for g in gaps if g > _TOL]
    if any(g < -_TOL for g in gaps):
        return None          # overlapping brackets
    if not pos:
        return 0.0
    if max(pos) - min(pos) > _TOL:
        return None
    return pos[0]


def _tiles(contracts: List[MarketContract]) -> bool:
    """True iff the (lo, hi) edges tile the whole real line with no gap or overlap."""
    edges = []
    for c in contracts:
        e = contract_edges(c)
        if e is None:
            return False
        edges.append(e)
    edges.sort(key=lambda e: (e[0] is not None, e[0] if e[0] is not None else 0.0))
    if edges[0][0] is not None or edges[-1][1] is not None:
        return False  # must be open below at the start and open above at the end
    for (_, hi), (lo, _) in zip(edges, edges[1:]):
        if hi is None or lo is None or abs(hi - lo) > _TOL:
            return False
    return True


def classify_kalshi_event(contracts: List[MarketContract],
                          event_mutually_exclusive: Optional[bool],
                          truncated: bool = False) -> None:
    """
    Mutates `contracts` in place. `event_mutually_exclusive` is the *event-level* flag from
    GET /events/{ticker} (market objects carry no such flag). `truncated` = the market list may be
    incomplete (page limit hit), which forbids claiming exhaustiveness.
    """
    if len(contracts) < 2:
        _set_all(contracts, EventKind.UNKNOWN)
        return

    types = {c.strike_type for c in contracts}

    if types == {"greater"} and all(c.floor_strike is not None for c in contracts) and not event_mutually_exclusive:
        _set_all(contracts, EventKind.CUMULATIVE_THRESHOLD)
        for c in contracts:
            c.model_edges = (c.floor_strike, None)
        return
    if types == {"less"} and all(c.cap_strike is not None for c in contracts) and not event_mutually_exclusive:
        _set_all(contracts, EventKind.CUMULATIVE_THRESHOLD)
        for c in contracts:
            c.model_edges = (None, c.cap_strike)
        return

    if not event_mutually_exclusive:
        _set_all(contracts, EventKind.UNKNOWN)
        return

    # Mutually exclusive event.
    _set_all(contracts, EventKind.PARTITION, exhaustive=False)
    if not types <= {"less", "between", "greater"}:
        return  # sports 1X2, custom strikes, ... : exclusivity known, exhaustiveness not provable here

    betweens = [c for c in contracts if c.strike_type == "between"]
    if any(c.floor_strike is None or c.cap_strike is None for c in betweens):
        return
    if any(c.strike_type == "less" and c.cap_strike is None for c in contracts):
        return
    if any(c.strike_type == "greater" and c.floor_strike is None for c in contracts):
        return

    step = _infer_step(betweens) if betweens else 0.0
    if step is None:
        return
    half = step / 2.0
    for c in contracts:
        c.resolution_step = step
        if c.strike_type == "less":          # X < cap   (strict)
            c.model_edges = (None, c.cap_strike - half)
        elif c.strike_type == "greater":     # X > floor (strict)
            c.model_edges = (c.floor_strike + half, None)
        else:                                # floor <= X <= cap, both inclusive on the print grid
            c.model_edges = (c.floor_strike - half, c.cap_strike + half)

    if _tiles(contracts) and not truncated:
        for c in contracts:
            c.exhaustive = True


# ----------------------------------------------------------------------------------------------
# Polymarket
# ----------------------------------------------------------------------------------------------
_ORDINAL = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(\+)?")


def _ordinal_edges(contracts: List[MarketContract]) -> bool:
    """
    Titles like '0 (0 bps)', '1 (25 bps)', ..., '12+ (300+ bps)' are ordered counts. Give them
    numeric edges (value +- step/2, open at both ends) so a distribution can be fitted.
    """
    vals = []
    for c in contracts:
        m = _ORDINAL.match(c.subtitle or "")
        if not m:
            return False
        vals.append((float(m.group(1)), bool(m.group(2))))
    xs = sorted(v for v, _ in vals)
    if len(set(xs)) != len(xs) or len(xs) < 3:
        return False
    steps = [b - a for a, b in zip(xs, xs[1:])]
    if max(steps) - min(steps) > _TOL:
        return False
    step = steps[0]
    for c, (v, _) in zip(contracts, vals):
        lo = None if v == xs[0] else v - step / 2.0
        hi = None if v == xs[-1] else v + step / 2.0
        c.model_edges = (lo, hi)
        c.resolution_step = step
    return True


def _stem(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(rf"\b(?:{_MONTHS})\b", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^a-z]+", " ", t)
    t = re.sub(r"\b(?:by|in|before|on|the|of)\b", " ", t)
    return " ".join(t.split())


def _is_nested_by_date(contracts: List[MarketContract]) -> bool:
    if len(contracts) < 2:
        return False
    ends = [c.expiration_time for c in contracts]
    if any(e is None for e in ends) or len(set(ends)) != len(ends):
        return False
    if len({_stem(c.title) for c in contracts}) != 1:
        return False
    # "on <date>" style markets are exclusive, not cumulative
    return not any(re.search(r"\bon\b", (c.title or "").lower()) for c in contracts)


def classify_polymarket_event(contracts: List[MarketContract], neg_risk: bool,
                              neg_risk_augmented: bool, n_unmapped: int = 0) -> None:
    """
    Mutates `contracts` (the event's *open* markets) in place.

    negRisk events are mutually exclusive. They are only treated as exhaustive when the event is not
    "augmented" (augmented events add placeholder outcomes like 'Party A' and an 'Other' bucket later)
    and every market could be mapped to a YES token.
    Non-negRisk events are only trusted as nested ladders when the markets are the same question
    asked for different deadlines; otherwise they are UNKNOWN (independent / unproven).
    """
    if len(contracts) < 2:
        _set_all(contracts, EventKind.UNKNOWN)
        return
    if neg_risk:
        _set_all(contracts, EventKind.PARTITION,
                 exhaustive=(not neg_risk_augmented and n_unmapped == 0))
        _ordinal_edges(contracts)
        return
    if _is_nested_by_date(contracts):
        _set_all(contracts, EventKind.NESTED_BY_DATE)
        return
    _set_all(contracts, EventKind.UNKNOWN)
