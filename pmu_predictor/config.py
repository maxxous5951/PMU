"""Réglages par défaut du modèle (poids en « unités de log-cote »).

Tout est modifiable depuis l'interface (curseurs) ou en éditant ce fichier.
Les poids par défaut sont des estimations de bon sens issues de la connaissance du trot :
ils sont faits pour être recalibrés avec `python -m pmu_predictor.backtest calibrer <dossier>`
dès que tu as plusieurs dizaines de courses avec leur résultat.
"""

from __future__ import annotations

# nom interne -> (poids par défaut, libellé affiché, aide)
FEATURES: dict[str, tuple[float, str, str]] = {
    "market":     (1.00, "Marché (cote finale)",
                   "Probabilité implicite de la cote : le meilleur indicateur isolé."),
    "steam":      (0.20, "Soutien du marché",
                   "Baisse de cote entre la cote de référence et la cote finale (argent tardif)."),
    "form":       (0.22, "Forme (musique)",
                   "Dernières performances pondérées par la récence."),
    "winrate":    (0.10, "Taux de victoires",
                   "Victoires / courses, ramené vers la moyenne pour les chevaux peu expérimentés."),
    "placerate":  (0.08, "Régularité (places)",
                   "(Victoires + places) / courses, ramené vers la moyenne."),
    "class":      (0.10, "Classe (gains/course)",
                   "Gains de carrière par course, comparés au reste du lot."),
    "yeargain":   (0.04, "Gains de l'année",
                   "Indicateur de forme financière sur l'année en cours."),
    "dq":         (1.00, "Risque de disqualification",
                   "Part de D (disqualifié) / T (tombé) dans les dernières sorties : pénalité."),
    "avis":       (0.15, "Avis de l'entraîneur",
                   "POSITIF = bonus, NEGATIF = malus."),
    "deferre":    (0.10, "Déferrage",
                   "Déferré des 4 pieds > antérieurs > postérieurs > protégé / ferré."),
    "handicap":   (0.25, "Handicap de distance",
                   "Pénalité par tranche de 25 m de recul par rapport au mieux placé."),
    "corde":      (0.20, "Place à la corde",
                   "Pénalité pour les numéros de corde très extérieurs (si l'info existe)."),
    "age":        (0.15, "Âge",
                   "Légère pénalité à partir de 8 ans."),
    "driver":     (0.12, "Driver / entraîneur",
                   "Bonus pour les équipes du haut de tableau (liste indicative ci-dessous)."),
    "discipline": (0.40, "Discipline (attelé / monté)",
                   "Pénalité si le cheval court rarement dans la discipline dominante de la course."),
    "pleine":     (0.50, "Jument pleine",
                   "Pénalité si la jument est annoncée pleine."),
}

DEFAULT_WEIGHTS: dict[str, float] = {k: v[0] for k, v in FEATURES.items()}

# Listes INDICATIVES (noms de famille, majuscules). À adapter selon la saison / l'hippodrome.
TOP_TIER = {
    "NIVARD", "ABRIVARD", "RAFFIN", "BAZIRE", "VERVA", "BEKAERT", "OUVRIE",
    "ROCHARD", "BARRIER",
}
SECOND_TIER = {
    "COLLETTE", "PLOQUIN", "BONNE", "LENAIN", "TRINQUART", "GIARD", "BOURGOIN",
    "LE BELLER", "DUVALDESTIN", "HALLAIS", "BRIDAULT", "THOMAS", "MARY",
}

# Constantes du modèle
PRIOR_WIN_RATE = 0.08       # ~ 1 chance sur 12
PRIOR_WIN_WEIGHT = 6.0
PRIOR_PLACE_RATE = 0.35
PRIOR_PLACE_WEIGHT = 6.0
Z_CLIP = 2.5
