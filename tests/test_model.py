"""Tests : lancer avec `python -m unittest discover -s tests` depuis la racine du dépôt."""

import copy
import unittest
from pathlib import Path

from pmu_predictor.loader import POST_RACE_KEYS, build_race, evaluate, load_race
from pmu_predictor.model import extract_features, predict
from pmu_predictor.musique import dq_rate, form_score, parse_musique

EX = Path(__file__).resolve().parent.parent / "data" / "exemples"


class MusiqueTests(unittest.TestCase):
    def test_parse(self):
        perfs = parse_musique("2a0a(25)Dm1a")
        self.assertEqual([p.place for p in perfs], ["2", "0", "D", "1"])
        self.assertEqual([p.discipline for p in perfs], ["a", "a", "m", "a"])
        self.assertTrue(perfs[2].new_year)

    def test_form_recent_counts_more(self):
        good_recent = form_score(parse_musique("1a1a0a0a0a0a"))
        good_old = form_score(parse_musique("0a0a0a0a1a1a"))
        self.assertGreater(good_recent, good_old)

    def test_dq(self):
        self.assertGreater(dq_rate(parse_musique("DaDa1a1a")), dq_rate(parse_musique("1a1a1aDa")))


class LeakageTests(unittest.TestCase):
    def test_post_race_fields_removed_and_result_kept_apart(self):
        lr = load_race(EX / "course1_dump.txt")
        self.assertIsNotNone(lr.result)
        self.assertEqual(lr.result[14], 1)  # n°14 a gagné
        for p in lr.race.participants:
            self.assertFalse(set(p) & POST_RACE_KEYS)

    def test_prediction_independent_of_result(self):
        lr = load_race(EX / "course1_dump.txt")
        base = [h.num for h in predict(lr.race).horses]
        raw = [dict(p, ordreArrivee=(i * 7) % 14 + 1, tempsObtenu=1, incident="X") for i, p in enumerate(lr.race.participants)]
        shuffled = build_race("x", raw)
        self.assertEqual([h.num for h in predict(shuffled.race).horses], base)

    def test_model_refuses_leaked_race(self):
        lr = load_race(EX / "course2.json")
        dirty = copy.deepcopy(lr.race)
        dirty.participants[0]["ordreArrivee"] = 1
        with self.assertRaises(ValueError):
            extract_features(dirty)


class LoaderTests(unittest.TestCase):
    def test_dump_and_json_have_expected_sizes(self):
        self.assertEqual(len(load_race(EX / "course1_dump.txt").race.participants), 14)
        self.assertEqual(len(load_race(EX / "course2.json").race.participants), 15)
        self.assertEqual(len(load_race(EX / "course3.json").race.participants), 12)

    def test_dump_odds_parsed(self):
        p = load_race(EX / "course1_dump.txt").race.participants[11]
        self.assertEqual(p["dernierRapportDirect"]["rapport"], 4.1)
        self.assertEqual(p["gainsParticipant"]["gainsCarriere"], 92268700)


class ModelTests(unittest.TestCase):
    def test_probabilities_sum_to_one_and_top5(self):
        for f in ("course1_dump.txt", "course2.json", "course3.json"):
            pred = predict(load_race(EX / f).race)
            self.assertAlmostEqual(sum(h.prob_model for h in pred.horses), 1.0, places=6)
            self.assertEqual(len(pred.top5), 5)

    def test_market_favourite_is_in_top3(self):
        for f in ("course1_dump.txt", "course2.json", "course3.json"):
            pred = predict(load_race(EX / f).race)
            fav = min((h for h in pred.horses if h.odds), key=lambda h: h.odds)
            self.assertLessEqual(fav.rank, 3)

    def test_evaluate(self):
        res = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        ev = evaluate([1, 2, 3, 6, 5], res)
        self.assertEqual(ev["bons_dans_top5"], 4)
        self.assertTrue(ev["tierce_ordre"])
        self.assertFalse(ev["quinte_desordre"])


if __name__ == "__main__":
    unittest.main()
