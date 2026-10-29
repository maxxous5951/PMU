"""Évaluation et calibration du modèle sur des courses passées.

Usage :
    python -m pmu_predictor.backtest evaluer  data/historique
    python -m pmu_predictor.backtest calibrer data/historique --sortie weights.json

Les fichiers doivent être des JSON PMU COMPLETS (avec `ordreArrivee`) : le résultat sert uniquement
à noter le pronostic, il n'est jamais fourni au modèle (voir loader.build_race).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from . import config as C
from .loader import LoadedRace, evaluate, load_race
from .model import extract_features, predict, score_features


def load_folder(folder: str | Path) -> list[LoadedRace]:
    races = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in (".json", ".txt"):
            continue
        try:
            lr = load_race(p)
        except Exception as exc:  # fichier illisible : on l'ignore mais on le dit
            print(f"  ignoré {p.name} : {exc}")
            continue
        if lr.result and len(lr.result) >= 3:
            races.append(lr)
        else:
            print(f"  ignoré {p.name} : pas de résultat exploitable")
    return races


def evaluer(races: list[LoadedRace], weights: dict[str, float] | None = None) -> dict[str, float]:
    n = len(races)
    agg = {"courses": n, "gagnant_trouve": 0, "gagnant_dans_top3_prono": 0, "tierce_desordre": 0,
           "quinte_desordre": 0, "quinte_ordre": 0, "bons_dans_top5_moy": 0.0,
           "positions_exactes_moy": 0.0}
    for lr in races:
        pred = predict(lr.race, weights)
        ev = evaluate([h.num for h in pred.top5], lr.result)
        for k in ("gagnant_trouve", "gagnant_dans_top3_prono", "tierce_desordre",
                  "quinte_desordre", "quinte_ordre"):
            agg[k] += int(ev[k])
        agg["bons_dans_top5_moy"] += ev["bons_dans_top5"]
        agg["positions_exactes_moy"] += ev["positions_exactes"]
    if n:
        agg["bons_dans_top5_moy"] /= n
        agg["positions_exactes_moy"] /= n
    return agg


# --------------------------------------------------------------------------- #
# Calibration : vraisemblance de Plackett-Luce sur les 5 premiers de chaque course
# --------------------------------------------------------------------------- #
def _prepare(races: list[LoadedRace]):
    data = []
    for lr in races:
        feats, _, _ = extract_features(lr.race)
        order = [n for n, _ in sorted(lr.result.items(), key=lambda kv: kv[1])][:5]
        idx = {h.num: i for i, h in enumerate(feats)}
        order_idx = [idx[n] for n in order if n in idx]
        if len(order_idx) >= 2:
            data.append((feats, order_idx))
    return data


def _loglik(data, weights) -> float:
    total, count = 0.0, 0
    for feats, order in data:
        scores = [s for s, _ in score_features(feats, weights)]
        remaining = list(range(len(scores)))
        for i in order:
            m = max(scores[j] for j in remaining)
            lse = m + math.log(sum(math.exp(scores[j] - m) for j in remaining))
            total += scores[i] - lse
            count += 1
            remaining.remove(i)
    return total / max(count, 1)


def calibrer(races: list[LoadedRace], iterations: int = 600, reg: float = 0.02,
             seed: int = 0) -> dict[str, float]:
    """Recherche aléatoire locale ; `reg` ramène les poids vers les valeurs par défaut (évite le surapprentissage)."""
    rng = random.Random(seed)
    data = _prepare(races)
    base = dict(C.DEFAULT_WEIGHTS)
    best = dict(base)

    def objective(w):
        return _loglik(data, w) - reg * sum((w[k] - base[k]) ** 2 for k in w)

    best_obj = objective(best)
    start_obj = best_obj
    for _ in range(iterations):
        cand = dict(best)
        for k in rng.sample(list(cand), k=min(3, len(cand))):
            cand[k] = max(0.0, cand[k] + rng.gauss(0, 0.12))
        obj = objective(cand)
        if obj > best_obj:
            best, best_obj = cand, obj
    print(f"Vraisemblance (moy. par place) : {start_obj:.4f} -> {best_obj:.4f} sur {len(data)} courses")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Évaluation / calibration sur courses passées")
    ap.add_argument("commande", choices=["evaluer", "calibrer"])
    ap.add_argument("dossier")
    ap.add_argument("--poids", help="fichier JSON de poids à utiliser pour l'évaluation")
    ap.add_argument("--sortie", default="weights.json", help="fichier de sortie pour la calibration")
    ap.add_argument("--iterations", type=int, default=600)
    args = ap.parse_args(argv)

    races = load_folder(args.dossier)
    if not races:
        raise SystemExit("Aucune course avec résultat trouvée.")

    if args.commande == "evaluer":
        w = json.loads(Path(args.poids).read_text()) if args.poids else None
        for k, v in evaluer(races, w).items():
            print(f"{k:28} {v:.3f}" if isinstance(v, float) else f"{k:28} {v}")
    else:
        if len(races) < 30:
            print(f"⚠ Seulement {len(races)} courses : la calibration risque de surapprendre (viser 100+).")
        w = calibrer(races, args.iterations)
        Path(args.sortie).write_text(json.dumps({k: round(v, 3) for k, v in w.items()}, indent=2))
        print(f"Poids enregistrés dans {args.sortie}")


if __name__ == "__main__":
    main()
