"""Lecture de la « musique » d'un cheval (suite de ses dernières performances).

Format : du plus récent au plus ancien, chaque course = place + discipline.
  "2a0a8a(25)2m1a"  ->  2e attelé, 0 (non placé) attelé, 8e attelé, [changement d'année], 2e monté, 1er attelé.
Place : 1..9, 0 = 10e ou plus ; D = disqualifié, A = arrêté, T = tombé, R = restée.
Discipline : a = attelé, m = monté, p = plat, h = haies, s = steeple, c = cross.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"\((\d{2})\)|([0-9DATRdatr])([amphsc])", re.I)

# Valeur d'une performance (1.0 = victoire)
PLACE_POINTS = {
    "1": 1.00, "2": 0.75, "3": 0.60, "4": 0.45, "5": 0.35,
    "6": 0.20, "7": 0.15, "8": 0.10, "9": 0.05, "0": 0.00,
    "D": -0.30, "A": -0.15, "T": -0.30, "R": -0.15,
}


@dataclass
class Perf:
    place: str       # "1".."9","0","D","A","T","R"
    discipline: str  # "a","m",...
    new_year: bool   # True si cette perf est la première après un marqueur d'année


def parse_musique(musique: str | None) -> list[Perf]:
    perfs: list[Perf] = []
    pending_year = False
    for m in _TOKEN.finditer(musique or ""):
        if m.group(1):
            pending_year = True
            continue
        perfs.append(Perf(m.group(2).upper(), m.group(3).lower(), pending_year))
        pending_year = False
    return perfs


def form_score(perfs: list[Perf], decay: float = 0.8, horizon: int = 8,
               prior: float = 0.25, prior_weight: float = 1.0) -> float:
    """Moyenne pondérée par la récence, ramenée vers un a priori si peu de courses."""
    num = den = 0.0
    for i, p in enumerate(perfs[:horizon]):
        w = decay ** i
        if p.new_year:
            w *= 0.85  # une perf de l'an dernier compte un peu moins
        num += w * PLACE_POINTS.get(p.place, 0.0)
        den += w
    return (num + prior * prior_weight) / (den + prior_weight)


def dq_rate(perfs: list[Perf], horizon: int = 6) -> float:
    """Part (pondérée) de disqualifications / chutes récentes (D, T) : risque d'allure irrégulière."""
    if not perfs:
        return 0.0
    num = den = 0.0
    for i, p in enumerate(perfs[:horizon]):
        w = 2.0 if i < 3 else 1.0
        den += w
        if p.place in ("D", "T"):
            num += w
    return num / den


def top3_recent(perfs: list[Perf], n: int = 5) -> int:
    return sum(1 for p in perfs[:n] if p.place in ("1", "2", "3"))


def last_place(perfs: list[Perf]) -> str | None:
    return perfs[0].place if perfs else None


def discipline_share(perfs: list[Perf], discipline: str, horizon: int = 8) -> float:
    sample = perfs[:horizon]
    if not sample:
        return 0.5
    return sum(1 for p in sample if p.discipline == discipline) / len(sample)
