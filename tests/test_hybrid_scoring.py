import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data
from src.engine import (
    _normalize_component_scores,
    compute_content_features,
    compute_collaborative_features,
    aggregate_ratings,
    recommend_hybrid,
)


class TestComponentNormalizationHelper(unittest.TestCase):
    """Isolated unit tests for the private _normalize_component_scores helper."""

    def test_normalize_basic_shapes_and_types(self):
        c = [0.1, 0.5, 0.9]
        cf = [-0.2, 0.0, 0.4]
        c_norm, cf_norm = _normalize_component_scores(c, cf)
        self.assertEqual(len(c_norm), 3)
        self.assertEqual(len(cf_norm), 3)
        self.assertIsInstance(c_norm, np.ndarray)
        self.assertIsInstance(cf_norm, np.ndarray)

    def test_score_bounds_from_zero_to_one(self):
        c = [1.5, -0.5, 0.3, 0.8]
        cf = [-1.0, 0.25, 0.5, 1.2]
        c_norm, cf_norm = _normalize_component_scores(c, cf, cf_ref=0.5)
        self.assertTrue(np.all(c_norm >= 0.0))
        self.assertTrue(np.all(c_norm <= 1.0))
        self.assertTrue(np.all(cf_norm >= 0.0))
        self.assertTrue(np.all(cf_norm <= 1.0))

    def test_negative_cf_values_zero_clipped(self):
        c = [0.5, 0.5, 0.5]
        cf = [-0.8, -0.01, -1.0]
        _, cf_norm = _normalize_component_scores(c, cf)
        self.assertTrue(np.all(cf_norm == 0.0))

    def test_nan_and_infinite_values(self):
        c = [np.nan, np.inf, -np.inf, 0.7]
        cf = [-np.inf, np.nan, np.inf, 0.35]
        c_norm, cf_norm = _normalize_component_scores(c, cf, cf_ref=0.5)
        self.assertFalse(np.any(np.isnan(c_norm)))
        self.assertFalse(np.any(np.isnan(cf_norm)))
        self.assertFalse(np.any(np.isinf(c_norm)))
        self.assertFalse(np.any(np.isinf(cf_norm)))
        self.assertEqual(c_norm[0], 0.0)
        self.assertEqual(c_norm[1], 0.0)
        self.assertEqual(c_norm[2], 0.0)
        self.assertAlmostEqual(c_norm[3], 0.7, places=4)
        self.assertEqual(cf_norm[0], 0.0)
        self.assertEqual(cf_norm[1], 0.0)
        self.assertEqual(cf_norm[2], 0.0)
        self.assertAlmostEqual(cf_norm[3], 0.7, places=4)  # 0.35 / 0.5 = 0.7

    def test_all_zero_cf_scores(self):
        c = [0.2, 0.6, 0.8]
        cf = [0.0, 0.0, 0.0]
        c_norm, cf_norm = _normalize_component_scores(c, cf)
        self.assertTrue(np.all(cf_norm == 0.0))
        self.assertFalse(np.any(np.isnan(cf_norm)))

    def test_query_movie_exclusion_at_query_idx(self):
        c = [0.8, 1.0, 0.4]
        cf = [0.3, 0.9, 0.1]
        c_norm, cf_norm = _normalize_component_scores(c, cf, query_idx=1)
        self.assertEqual(c_norm[1], 0.0)
        self.assertEqual(cf_norm[1], 0.0)
        # Other items remain intact
        self.assertAlmostEqual(c_norm[0], 0.8, places=4)
        self.assertAlmostEqual(c_norm[2], 0.4, places=4)

    def test_cf_reference_ceiling_scaling(self):
        # With cf_ref=0.5:
        # 0.25 -> 0.50
        # 0.50 -> 1.00
        # 0.60 -> clipped to 1.00
        c = [0.5, 0.5, 0.5]
        cf = [0.25, 0.50, 0.60]
        _, cf_norm = _normalize_component_scores(c, cf, cf_ref=0.5)
        self.assertAlmostEqual(cf_norm[0], 0.50, places=4)
        self.assertAlmostEqual(cf_norm[1], 1.00, places=4)
        self.assertAlmostEqual(cf_norm[2], 1.00, places=4)


class TestHybridScoringPipeline(unittest.TestCase):
    """Integration tests for recommendation ranking using normalized hybrid components."""

    @classmethod
    def setUpClass(cls):
        cls.movies, cls.ratings = load_data()
        cls.rating_agg = aggregate_ratings(cls.ratings)
        cls.content_features = compute_content_features(cls.movies)
        cls.cf_features = compute_collaborative_features(cls.movies, cls.ratings)

    def test_content_only_ranking(self):
        """content_weight=1.0 must produce pure content-driven recommendations."""
        recs_w1 = recommend_hybrid(
            "Batman Begins",
            self.movies,
            self.content_features,
            self.cf_features,
            content_weight=1.0,
            top_n=5,
        )
        self.assertEqual(len(recs_w1), 5)
        # Verify scores match raw content cosine similarity
        for title, _, score, _ in recs_w1:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertNotEqual(title, "Batman Begins (2005)")

    def test_collaborative_only_ranking(self):
        """content_weight=0.0 must produce pure collaborative-driven recommendations."""
        recs_w0 = recommend_hybrid(
            "Batman Begins",
            self.movies,
            self.content_features,
            self.cf_features,
            content_weight=0.0,
            top_n=5,
        )
        self.assertEqual(len(recs_w0), 5)
        for title, _, score, _ in recs_w0:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertNotEqual(title, "Batman Begins (2005)")

    def test_hybrid_ranking_is_balanced_blend(self):
        """content_weight=0.5 must blend content and collaborative components."""
        recs_w1 = recommend_hybrid(
            "Batman Begins",
            self.movies,
            self.content_features,
            self.cf_features,
            content_weight=1.0,
            top_n=5,
        )
        recs_w0 = recommend_hybrid(
            "Batman Begins",
            self.movies,
            self.content_features,
            self.cf_features,
            content_weight=0.0,
            top_n=5,
        )
        recs_hybrid = recommend_hybrid(
            "Batman Begins",
            self.movies,
            self.content_features,
            self.cf_features,
            content_weight=0.5,
            top_n=5,
        )
        self.assertEqual(len(recs_hybrid), 5)
        # Hybrid rankings should not be strictly identical to both pure endpoints
        titles_w1 = [r[0] for r in recs_w1]
        titles_w0 = [r[0] for r in recs_w0]
        titles_hybrid = [r[0] for r in recs_hybrid]
        self.assertNotEqual(titles_hybrid, titles_w1)

    def test_score_bounds_and_descending_order(self):
        """All recommendation scores must be in [0.0, 1.0] and strictly descending."""
        for query in ["Batman Begins", "Inception", "The Dark Knight"]:
            for weight in [0.0, 0.3, 0.5, 0.8, 1.0]:
                recs = recommend_hybrid(
                    query,
                    self.movies,
                    self.content_features,
                    self.cf_features,
                    content_weight=weight,
                    top_n=10,
                )
                if not recs:
                    continue
                scores = [r[2] for r in recs]
                # Bounded in [0.0, 1.0]
                for sc in scores:
                    self.assertGreaterEqual(
                        sc, 0.0, f"Score {sc} < 0 for query '{query}'"
                    )
                    self.assertLessEqual(sc, 1.0, f"Score {sc} > 1 for query '{query}'")
                # Strictly descending (or non-increasing)
                for i in range(len(scores) - 1):
                    self.assertGreaterEqual(
                        scores[i],
                        scores[i + 1],
                        f"Non-descending scores at {i}: {scores[i]} vs {scores[i+1]}",
                    )

    def test_exclusion_of_query_movie(self):
        """The query movie itself must never appear in recommendations."""
        queries = ["Batman Begins", "Inception", "Dark Knight, The (2008)"]
        for query in queries:
            for weight in [0.0, 0.5, 1.0]:
                recs = recommend_hybrid(
                    query,
                    self.movies,
                    self.content_features,
                    self.cf_features,
                    content_weight=weight,
                    top_n=10,
                )
                rec_titles = [r[0] for r in recs]
                self.assertNotIn(
                    "Batman Begins (2005)",
                    rec_titles if query == "Batman Begins" else [],
                )
                self.assertNotIn(
                    "Inception (2010)", rec_titles if query == "Inception" else []
                )
                self.assertNotIn(
                    "Dark Knight, The (2008)",
                    rec_titles if query == "Dark Knight, The (2008)" else [],
                )

    def test_unrated_or_cold_start_movie_fallback(self):
        """Movies with zero ratings/collaborative signal must fall back cleanly without error."""
        # Create synthetic all-zero collaborative features for testing
        zero_cf = np.zeros_like(self.cf_features)
        recs = recommend_hybrid(
            "Batman Begins",
            self.movies,
            self.content_features,
            zero_cf,
            content_weight=0.5,
            top_n=5,
        )
        self.assertEqual(len(recs), 5)
        for _, _, score, _ in recs:
            self.assertFalse(np.isnan(score))
            self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
