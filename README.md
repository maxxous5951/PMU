# PMU Predictor

Application Python (interface graphique Tkinter, **aucune dépendance externe**) qui charge des fichiers JSON de course PMU et produit un pronostic (top 5 + outsider) **uniquement à partir des informations d'avant-course**.

## Lancer

```bash
python main.py                  # interface graphique
python main.py course.json      # pronostic en ligne de commande
```

Tkinter est fourni avec Python sous Windows et macOS. Sous Linux : `sudo apt install python3-tk`.

## Formats acceptés

- JSON PMU (`{"participants": [...]}` ou liste de participants) ;
- copier-coller « arbre » des outils développeur du navigateur (lignes `clé<TAB>valeur`).

## Pas de triche : séparation avant / après course

`loader.build_race` retire dès le chargement tous les champs d'après-course (`ordreArrivee`, `tempsObtenu`, `reductionKilometrique`, `commentaireApresCourse`, `incident`…). Le modèle refuse d'ailleurs toute course qui en contiendrait encore. Le résultat est gardé à part et ne sert qu'au bouton **« Vérifier avec le résultat réel »**, après la prédiction.

Pour renseigner une arrivée : bouton **« Saisir l'arrivée… »** (ex. `7-9-2-3-13`), ce qui crée un fichier `<course>.arrivee` à côté du JSON.

## Ce que le modèle regarde

Cote finale (probabilité implicite) et son évolution, musique pondérée par la récence, pénalité de disqualification, taux de victoires et de places (ramenés vers la moyenne pour les chevaux peu expérimentés), gains par course, avis de l'entraîneur, déferrage, handicap de distance, place à la corde, âge, discipline (attelé/monté), driver/entraîneur (liste indicative dans `config.py`), jument pleine.

Chaque critère est transformé en score, pondéré (curseurs dans l'interface), puis converti en probabilités. Le détail par cheval montre ce qui pèse pour ou contre lui.

## Calibrer sur tes courses passées

Mets des JSON de courses terminées (avec `ordreArrivee`, ou un fichier `.arrivee` à côté) dans un dossier :

```bash
python -m pmu_predictor.backtest evaluer  data/historique
python -m pmu_predictor.backtest calibrer data/historique --sortie weights.json
```

Le fichier `weights.json` se recharge dans l'interface (« Charger poids… »). Compter au moins 100 courses avant de se fier à la calibration.

## Tests

```bash
python -m unittest discover -s tests
```

## Limites

Les courses hippiques restent très aléatoires et la cote finale est déjà un excellent indicateur : le modèle vise la régularité et l'explication, pas un gain garanti. Jouer comporte des risques.
