"""Modèle de pronostic : score = somme pondérée de critères, converti en probabilités (softmax).

Le modèle ne reçoit qu'un objet `Race` déjà nettoyé des données d'après-course (voir loader.py).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

from . import config as C
from .loader import POST_RACE_KEYS, Race
from .musique import (Perf, discipline_share, dq_rate, form_score,
                      last_place, parse_musique)


# --------------------------------------------------------------------------- #
# Structures
# --------------------------------------------------------------------------- #
@dataclass
class HorseFeatures:
    num: int
    nom: str
    driver: str
    entraineur: str
    odds: float | None
    musique: str
    values: dict[str, float]      # valeur de chaque critère (z-score ou valeur brute)


@dataclass
class HorseScore:
    num: int
    nom: str
    driver: str
    odds: float | None
    musique: str
    score: float
    prob_model: float
    prob_market: float | None
    value_ratio: float | None             # prob_model / prob_market
    contributions: dict[str, float]
    rank: int = 0

    def reasons(self, n: int = 3) -> tuple[list[str], list[str]]:
        items = sorted(self.contributions.items(), key=lambda kv: kv[1], reverse=True)
        pos = [f"{C.FEATURES[k][1]} ({v:+.2f})" for k, v in items if v > 0.05 and k != "market"][:n]
        neg = [f"{C.FEATURES[k][1]} ({v:+.2f})" for k, v in reversed(items) if v < -0.05 and k != "market"][:n]
        return pos, neg


@dataclass
class Prediction:
    race_name: str
    horses: list[HorseScore]              # triés du meilleur au moins bon
    top5: list[HorseScore]
    outsider: HorseScore | None
    openness: str                         # "fermée" / "moyenne" / "ouverte"
    has_market: bool
    discipline: str
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction des critères
# --------------------------------------------------------------------------- #
def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _odds(p: dict[str, Any]) -> tuple[float | None, float | None]:
    def get(key: str) -> float | None:
        r = p.get(key) or {}
        v = r.get("rapport") if isinstance(r, dict) else None
        v = _num(v, 0.0)
        return v if v > 1.0 else None
    return get("dernierRapportDirect"), get("dernierRapportReference")


def _surname_tier(name: str) -> float:
    n = re.sub(r"[.\-]", " ", (name or "").upper())
    n = re.sub(r"\s+", " ", n).strip()
    for s in C.TOP_TIER:
        if re.search(rf"\b{re.escape(s)}\b", n):
            return 1.0
    for s in C.SECOND_TIER:
        if re.search(rf"\b{re.escape(s)}\b", n):
            return 0.5
    return 0.0


def _deferre(value: str | None) -> float:
    v = (value or "").upper()
    if v.startswith("DEFERRE_ANTERIEURS_POSTERIEURS"):
        return 1.0
    if v.startswith("DEFERRE_ANTERIEURS"):
        return 0.7
    if "DEFERRE_POSTERIEURS" in v or "DEFERRRE_POSTERIEURS" in v:  # "DEFERRRE" : faute de frappe présente dans les données PMU
        return 0.4
    return 0.0


def _age_penalty(age: Any) -> float:
    a = _num(age, 5)
    if a <= 7:
        return 0.0
    return -min(1.0, (a - 7) * 0.25)


def _zscores(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    m, s = mean(values), pstdev(values)
    if s < 1e-9:
        return [0.0] * len(values)
    return [max(-C.Z_CLIP, min(C.Z_CLIP, (v - m) / s)) for v in values]


def race_discipline(parsed: list[list[Perf]]) -> str:
    """Discipline dominante de la course, déduite des dernières sorties du lot (a = attelé, m = monté)."""
    counts: dict[str, int] = {}
    for perfs in parsed:
        for p in perfs[:3]:
            counts[p.discipline] = counts.get(p.discipline, 0) + 1
    return max(counts, key=counts.get) if counts else "a"


def extract_features(race: Race) -> tuple[list[HorseFeatures], str, list[str]]:
    warnings: list[str] = []
    runners = [p for p in race.participants if str(p.get("statut", "PARTANT")).upper() == "PARTANT"]
    if len(runners) < 2:
        raise ValueError("Il faut au moins 2 partants pour établir un pronostic.")

    # Garde anti-fuite : aucun champ d'après-course ne doit subsister.
    leaked = {k for p in runners for k in p if k in POST_RACE_KEYS}
    if leaked:
        raise ValueError(f"Données d'après-course détectées ({sorted(leaked)}) : utiliser loader.load_race().")

    parsed = [parse_musique(p.get("musique")) for p in runners]
    disc = race_discipline(parsed)

    # --- marché
    finals, refs = zip(*[_odds(p) for p in runners])
    odds = [f if f is not None else r for f, r in zip(finals, refs)]
    have = [o for o in odds if o]
    has_market = len(have) >= max(2, len(runners) // 2)
    if not has_market:
        warnings.append("Cotes absentes ou incomplètes : le pronostic repose uniquement sur les autres critères.")
    inv = [1.0 / o if o else None for o in odds]
    total = sum(x for x in inv if x) or 1.0
    min_p = min((x / total for x in inv if x), default=0.01) / 2
    p_market = [(x / total) if x else min_p for x in inv]
    market = [math.log(p) if has_market else 0.0 for p in p_market]

    steam = []
    for f, r in zip(finals, refs):
        steam.append(max(-1.0, min(1.0, math.log(r / f))) if f and r else 0.0)

    # --- historique
    form_raw = [form_score(pf) for pf in parsed]
    win_raw, place_raw, class_raw, year_raw = [], [], [], []
    for p in runners:
        n = _num(p.get("nombreCourses"))
        v = _num(p.get("nombreVictoires"))
        pl = _num(p.get("nombrePlaces"))
        win_raw.append((v + C.PRIOR_WIN_RATE * C.PRIOR_WIN_WEIGHT) / (n + C.PRIOR_WIN_WEIGHT))
        place_raw.append((v + pl + C.PRIOR_PLACE_RATE * C.PRIOR_PLACE_WEIGHT) / (n + C.PRIOR_PLACE_WEIGHT))
        g = p.get("gainsParticipant") or {}
        gains = _num(g.get("gainsCarriere"))
        class_raw.append(math.log1p(gains / max(n, 1.0) / 100.0))
        year_raw.append(math.log1p(_num(g.get("gainsAnneeEnCours")) / 100.0))
    z_form, z_win, z_place = _zscores(form_raw), _zscores(win_raw), _zscores(place_raw)
    z_class, z_year = _zscores(class_raw), _zscores(year_raw)

    # --- handicap & corde
    handis = [_num(p.get("handicapDistance")) for p in runners]
    hmin = min(handis) if any(handis) else 0.0

    feats: list[HorseFeatures] = []
    for i, p in enumerate(runners):
        avis = {"POSITIF": 1.0, "NEGATIF": -1.0}.get(str(p.get("avisEntraineur", "")).upper(), 0.0)
        handicap = -((handis[i] - hmin) / 25.0) if handis[i] else 0.0
        corde_n = _num(p.get("placeCorde"), 0.0)
        corde = -min(1.0, max(0.0, corde_n - 8) / 7.0) if corde_n else 0.0
        discipline = (discipline_share(parsed[i], disc) - 1.0) if len(parsed[i]) >= 3 else 0.0
        driver_tier = max(_surname_tier(p.get("driver", "")), _surname_tier(p.get("entraineur", "")))
        values = {
            "market": market[i],
            "steam": steam[i],
            "form": z_form[i],
            "winrate": z_win[i],
            "placerate": z_place[i],
            "class": z_class[i],
            "yeargain": z_year[i],
            "dq": -dq_rate(parsed[i]),
            "avis": avis,
            "deferre": _deferre(p.get("deferre")),
            "handicap": handicap,
            "corde": corde,
            "age": _age_penalty(p.get("age")),
            "driver": driver_tier,
            "discipline": discipline,
            "pleine": -1.0 if p.get("jumentPleine") else 0.0,
        }
        feats.append(HorseFeatures(
            num=int(p.get("numPmu", i + 1)), nom=str(p.get("nom", f"N°{i+1}")),
            driver=str(p.get("driver", "")), entraineur=str(p.get("entraineur", "")),
            odds=odds[i], musique=str(p.get("musique", "")), values=values))
    return feats, disc, warnings


# --------------------------------------------------------------------------- #
# Score et probabilités
# --------------------------------------------------------------------------- #
def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def score_features(feats: list[HorseFeatures], weights: dict[str, float]) -> list[tuple[float, dict[str, float]]]:
    out = []
    for h in feats:
        contrib = {k: weights.get(k, 0.0) * h.values.get(k, 0.0) for k in C.FEATURES}
        out.append((sum(contrib.values()), contrib))
    return out


def predict(race: Race, weights: dict[str, float] | None = None) -> Prediction:
    w = {**C.DEFAULT_WEIGHTS, **(weights or {})}
    feats, disc, warnings = extract_features(race)
    scored = score_features(feats, w)
    probs = _softmax([s for s, _ in scored])

    inv = [1.0 / h.odds if h.odds else None for h in feats]
    tot = sum(x for x in inv if x)
    horses: list[HorseScore] = []
    for h, (s, contrib), pm in zip(feats, scored, probs):
        pk = (1.0 / h.odds) / tot if (h.odds and tot) else None
        horses.append(HorseScore(
            num=h.num, nom=h.nom, driver=h.driver, odds=h.odds, musique=h.musique,
            score=s, prob_model=pm, prob_market=pk,
            value_ratio=(pm / pk) if pk else None, contributions=contrib))
    horses.sort(key=lambda x: x.score, reverse=True)
    for i, h in enumerate(horses, 1):
        h.rank = i

    top5 = horses[:5]
    outsider = None
    cands = [h for h in horses[5:10] if h.odds and h.odds >= 12 and h.value_ratio and h.value_ratio > 1.15]
    if cands:
        outsider = max(cands, key=lambda h: h.value_ratio)

    p1 = horses[0].prob_model
    openness = "fermée" if p1 >= 0.35 else "ouverte" if p1 < 0.20 else "moyenne"
    if len(horses) >= 14:
        warnings.append("Grand champ (14+ partants) : l'ordre exact est très difficile à prévoir.")
    return Prediction(race.name, horses, top5, outsider, openness,
                      any(h.odds for h in horses), disc, warnings)


def format_prediction(pred: Prediction) -> str:
    lines = [f"Pronostic — {pred.race_name}  (course {pred.openness}, discipline dominante : "
             f"{'attelé' if pred.discipline == 'a' else 'monté' if pred.discipline == 'm' else pred.discipline})", ""]
    for h in pred.top5:
        cote = f"{h.odds:g}" if h.odds else "n/c"
        pos, neg = h.reasons(2)
        why = "; ".join(pos) if pos else "—"
        lines.append(f" {h.rank}. n°{h.num:<2} {h.nom:<22} cote {cote:<6} proba {h.prob_model:5.1%}   + {why}")
    if pred.outsider:
        o = pred.outsider
        lines.append(f"\nOutsider à surveiller : n°{o.num} {o.nom} (cote {o.odds:g}, "
                     f"modèle {o.prob_model:.1%} vs marché {o.prob_market:.1%})")
    lines += [f"\n⚠ {w}" for w in pred.warnings]
    return "\n".join(lines)
