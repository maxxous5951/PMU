"""Chargement des fichiers de course PMU.

Règle d'or : le modèle ne doit JAMAIS voir une information postérieure à la course.
`load_race` renvoie donc deux objets séparés :
  - `Race`   : uniquement des données d'avant-course (passées au modèle) ;
  - `result` : l'ordre d'arrivée éventuel (jamais passé au modèle, sert à vérifier après coup).

Formats acceptés :
  - JSON PMU (API) : {"participants": [...]} ou une simple liste de participants ;
  - copier-coller "arbre" des outils développeur (lignes `clé<TAB>valeur`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Champs connus pour n'exister qu'APRÈS la course : supprimés dès le chargement.
POST_RACE_KEYS = {
    "ordreArrivee",
    "tempsObtenu",
    "reductionKilometrique",
    "commentaireApresCourse",
    "incident",
    "distanceChevalPrecedent",
    "rapportDefinitif",
    "gainsCourse",
    "dernierRapportDefinitif",
}


@dataclass
class Race:
    name: str
    participants: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedRace:
    race: Race
    result: dict[int, int] | None  # numPmu -> place d'arrivée (None si absent)
    removed_fields: set[str]


# --------------------------------------------------------------------------- #
# Lecture du format "dump" (copier-coller des outils développeur)
# --------------------------------------------------------------------------- #
_NESTED_GROUPS = {
    "robe": {"code", "libelleCourt", "libelleLong"},
    "gainsParticipant": {
        "gainsCarriere", "gainsVictoires", "gainsPlace",
        "gainsAnneeEnCours", "gainsAnneePrecedente",
    },
    "dernierRapportDirect": {
        "typePari", "rapport", "typeRapport", "nombreIndicateurTendance",
        "dateRapport", "favoris", "numPmu1", "grossePrise",
    },
    "dernierRapportReference": {
        "typePari", "rapport", "typeRapport", "nombreIndicateurTendance",
        "dateRapport", "favoris", "numPmu1", "grossePrise",
    },
    "commentaireApresCourse": {"texte", "source"},
}


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return None
    raw = re.sub(r"JS:[-0-9.eE]+$", "", raw)  # "26.0JS:26" -> "26.0"
    if raw in ("true", "false"):
        return raw == "true"
    if raw == "null":
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_dump(text: str) -> dict[str, Any]:
    """Convertit le format `clé<TAB>valeur` en dictionnaire {"participants": [...]}."""
    participants: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    group: str | None = None
    in_participants = False

    for line in text.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition("\t")
        key = key.strip()
        value = value.strip()

        if key == "participants" and value == "":
            in_participants = True
            continue
        if not in_participants:
            continue
        if key in ("ecuries", "spriteCasaques", "cached"):
            break

        if key.isdigit() and value == "":
            current = {}
            participants.append(current)
            group = None
            continue
        if current is None:
            continue

        if value == "" and key in _NESTED_GROUPS:
            group = key
            current[key] = {}
            continue

        if group is not None and key in _NESTED_GROUPS[group]:
            current[group][key] = _parse_scalar(value)
            continue

        group = None
        current[key] = _parse_scalar(value)

    return {"participants": participants}


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #
def _extract_participants(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("participants", "partants"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Aucune liste de participants trouvée dans le fichier.")


def parse_arrivee(text: str) -> dict[int, int]:
    """"7-9-2-3-13" -> {7: 1, 9: 2, 2: 3, 3: 4, 13: 5}."""
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) < 3 or len(set(nums)) != len(nums):
        raise ValueError("Arrivée invalide : donner au moins 3 numéros distincts, ex. 7-9-2-3-13")
    return {n: i + 1 for i, n in enumerate(nums)}


def arrivee_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".arrivee")


def load_race(path: str | Path) -> LoadedRace:
    """Charge une course. Si `<fichier>.arrivee` existe (ex. contenu « 7-9-2-3-13 »), il fournit le résultat
    pour la vérification et la calibration, sans jamais être transmis au modèle."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        data = json.loads(text)
    else:
        data = parse_dump(text)
    lr = build_race(path.stem, _extract_participants(data))
    side = arrivee_path(path)
    if lr.result is None and side.exists():
        lr.result = parse_arrivee(side.read_text(encoding="utf-8"))
    return lr


def build_race(name: str, raw_participants: list[dict[str, Any]]) -> LoadedRace:
    result: dict[int, int] = {}
    removed: set[str] = set()
    clean: list[dict[str, Any]] = []

    for p in raw_participants:
        num = p.get("numPmu")
        arrivee = p.get("ordreArrivee")
        if isinstance(num, int) and isinstance(arrivee, int):
            result[num] = arrivee
        removed |= {k for k in p if k in POST_RACE_KEYS}
        clean.append({k: v for k, v in p.items() if k not in POST_RACE_KEYS})

    return LoadedRace(
        race=Race(name=name, participants=clean),
        result=result or None,
        removed_fields=removed,
    )


def evaluate(prediction_numbers: list[int], result: dict[int, int]) -> dict[str, Any]:
    """Compare un pronostic (liste de numPmu, du meilleur au moins bon) au résultat réel."""
    finish = sorted(result.items(), key=lambda kv: kv[1])
    top5 = [n for n, _ in finish[:5]]
    pred5 = prediction_numbers[:5]
    return {
        "arrivee_top5": top5,
        "pronostic_top5": pred5,
        "bons_dans_top5": len(set(pred5) & set(top5)),
        "gagnant_trouve": bool(pred5) and bool(top5) and pred5[0] == top5[0],
        "gagnant_dans_top3_prono": bool(top5) and top5[0] in pred5[:3],
        "tierce_ordre": pred5[:3] == top5[:3],
        "tierce_desordre": len(top5) >= 3 and set(pred5[:3]) == set(top5[:3]),
        "quinte_ordre": pred5 == top5,
        "quinte_desordre": len(top5) == 5 and set(pred5) == set(top5),
        "positions_exactes": sum(1 for a, b in zip(pred5, top5) if a == b),
    }
