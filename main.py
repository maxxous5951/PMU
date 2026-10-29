"""Point d'entrée.

    python main.py                       -> interface graphique
    python main.py course1.json ...      -> pronostic en ligne de commande (sans interface)
"""

import sys

from pmu_predictor.loader import load_race
from pmu_predictor.model import format_prediction, predict


def main() -> None:
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not files:
        from pmu_predictor.gui import run
        run()
        return
    for path in files:
        print(format_prediction(predict(load_race(path).race)))
        print("-" * 78)


if __name__ == "__main__":
    main()
