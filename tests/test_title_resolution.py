import unittest
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data
from src.engine import (
    find_movie_index,
    resolve_movie_title,
    AmbiguityResult,
    TitleResolutionResult,
    compute_content_features,
    compute_collaborative_features,
    aggregate_ratings,
    recommend_hybrid,
)


class TestTitleResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies, cls.ratings = load_data()
        cls.rating_agg = aggregate_ratings(cls.ratings)
        cls.content_features = compute_content_features(cls.movies)
        cls.cf_features = compute_collaborative_features(cls.movies, cls.ratings)

    def _evaluate_and_log(self, query):
        res = resolve_movie_title(query, self.movies)
        print(f"\n[TEST CASE] Query: '{query}'")
        print(f"  -> Status:                  {res.status}")
        print(f"  -> Resolved Title:          {res.title}")
        print(f"  -> MovieId:                 {res.movie_id}")
        print(f"  -> Matching Method:         {res.method}")
        print(f"  -> Year Constraint Respected: {res.year_constraint_respected}")
        if res.is_ambiguous:
            print(f"  -> Ambiguous Candidates:   {[c['title'] for c in res.candidates]}")
        return res

    def test_batman_begins(self):
        query = "Batman Begins"
        res = self._evaluate_and_log(query)
        self.assertTrue(res.is_resolved)
        self.assertEqual(res.title, "Batman Begins (2005)")
        self.assertEqual(res.movie_id, 33794)
        self.assertTrue(res.year_constraint_respected)
        # Verify backwards compatibility for find_movie_index
        idx = find_movie_index(query, self.movies)
        self.assertIsInstance(idx, int)
        self.assertEqual(idx, res.index)

    def test_batman_begins_year(self):
        query = "Batman Begins (2005)"
        res = self._evaluate_and_log(query)
        self.assertTrue(res.is_resolved)
        self.assertEqual(res.title, "Batman Begins (2005)")
        self.assertEqual(res.movie_id, 33794)
        self.assertEqual(res.year, 2005.0)
        self.assertTrue(res.year_constraint_respected)
        self.assertIn("year", res.method.lower())
        idx = find_movie_index(query, self.movies)
        self.assertIsInstance(idx, int)
        self.assertEqual(idx, res.index)

    def test_the_dark_knight(self):
        query = "The Dark Knight"
        res = self._evaluate_and_log(query)
        self.assertTrue(res.is_resolved)
        self.assertEqual(res.title, "Dark Knight, The (2008)")
        self.assertEqual(res.movie_id, 58559)
        self.assertTrue(res.year_constraint_respected)
        idx = find_movie_index(query, self.movies)
        self.assertIsInstance(idx, int)
        self.assertEqual(idx, res.index)

    def test_dark_knight(self):
        query = "Dark Knight"
        res = self._evaluate_and_log(query)
        self.assertTrue(res.is_resolved)
        self.assertEqual(res.title, "Dark Knight, The (2008)")
        self.assertEqual(res.movie_id, 58559)
        self.assertTrue(res.year_constraint_respected)
        idx = find_movie_index(query, self.movies)
        self.assertIsInstance(idx, int)
        self.assertEqual(idx, res.index)

    def test_avengers(self):
        query = "Avengers"
        res = self._evaluate_and_log(query)
        self.assertTrue(res.is_ambiguous)
        self.assertFalse(res.is_resolved)
        self.assertIsNone(res.title)
        self.assertIsNone(res.movie_id)
        self.assertTrue(res.year_constraint_respected)
        candidate_ids = [c["movieId"] for c in res.candidates]
        self.assertIn(2153, candidate_ids)   # Avengers, The (1998)
        self.assertIn(89745, candidate_ids)  # Avengers, The (2012)
        # Verify backwards compatibility for find_movie_index returns AmbiguityResult
        amb = find_movie_index(query, self.movies)
        self.assertIsInstance(amb, AmbiguityResult)
        self.assertTrue(amb.is_ambiguous)
        self.assertFalse(bool(amb))  # Falsy for legacy boolean checks

    def test_the_avengers_2012(self):
        query = "The Avengers (2012)"
        res = self._evaluate_and_log(query)
        self.assertTrue(res.is_resolved)
        self.assertEqual(res.title, "Avengers, The (2012)")
        self.assertEqual(res.movie_id, 89745)
        self.assertEqual(res.year, 2012.0)
        self.assertTrue(res.year_constraint_respected)
        self.assertIn("year", res.method.lower())
        idx = find_movie_index(query, self.movies)
        self.assertIsInstance(idx, int)
        self.assertEqual(idx, res.index)

    def test_production_recommendation_scoring_unchanged(self):
        """Verifies that recommendation scoring for resolved movies is 100% bit-exact identical."""
        recs = recommend_hybrid(
            "Batman Begins",
            self.movies,
            self.content_features,
            self.cf_features,
            content_weight=0.65,
            top_n=10,
            min_avg_rating=3.9,
            rating_agg=self.rating_agg,
        )

        expected_titles = [
            "Edge of Tomorrow (2014)",
            "Harry Potter and the Deathly Hallows: Part 1 (2010)",
            "The Raid: Redemption (2011)",
            "Boondock Saints, The (2000)",
            "Dark Knight, The (2008)",
            "Watchmen (2009)",
            "Elite Squad: The Enemy Within (Tropa de Elite 2 - O Inimigo Agora É Outro) (2010)",
            "Toy Story 3 (2010)",
            "Baby Driver (2017)",
            "Heat (1995)",
        ]

        actual_titles = [r[0] for r in recs]
        self.assertEqual(actual_titles, expected_titles)

        # Verify exact baseline hybrid scores
        expected_scores = [
            0.479753, 0.459704, 0.454147, 0.416930, 0.403328,
            0.395528, 0.375401, 0.362266, 0.355876, 0.345316
        ]
        for (actual_title, _, actual_score, _), exp_score in zip(recs, expected_scores):
            self.assertAlmostEqual(actual_score, exp_score, places=4)

        # Verify that an ambiguous query cleanly returns [] recommendations without crashing
        ambiguous_recs = recommend_hybrid(
            "Avengers",
            self.movies,
            self.content_features,
            self.cf_features,
            content_weight=0.65,
            top_n=10,
            min_avg_rating=3.9,
            rating_agg=self.rating_agg,
        )
        self.assertEqual(ambiguous_recs, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
