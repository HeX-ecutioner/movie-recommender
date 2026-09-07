import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.index import get_recommendations, movies
from src.engine import resolve_movie_title
from src.utils import fix_title_display


class TestTitleDisplayFormatting(unittest.TestCase):
    """Verifies that fix_title_display accurately, safely, and idempotently formats movie titles."""

    def test_regression_formatting_cases(self):
        cases = [
            ("Avengers, The (1998)", "The Avengers (1998)"),
            ("Avengers, The (2012)", "The Avengers (2012)"),
            ("Dark Knight, The (2008)", "The Dark Knight (2008)"),
            ("Lives of Others, The (2006)", "The Lives of Others (2006)"),
            ("Toy Story (1995)", "Toy Story (1995)"),
            ("Movie, Title (2020)", "Movie, Title (2020)"),
            ("The Avengers (2012)", "The Avengers (2012)"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                actual = fix_title_display(raw)
                self.assertEqual(actual, expected)

    def test_idempotence_double_formatting(self):
        raw = "The Avengers (2012)"
        once = fix_title_display(raw)
        twice = fix_title_display(once)
        self.assertEqual(once, "The Avengers (2012)")
        self.assertEqual(twice, "The Avengers (2012)")

        raw_inv = "Avengers, The (2012)"
        once_inv = fix_title_display(raw_inv)
        twice_inv = fix_title_display(once_inv)
        self.assertEqual(once_inv, "The Avengers (2012)")
        self.assertEqual(twice_inv, "The Avengers (2012)")

    def test_additional_article_variants_and_edge_cases(self):
        cases = [
            ("Something, A (2010)", "A Something (2010)"),
            ("Elephant, An (2015)", "An Elephant (2015)"),
            ("Avengers, The", "The Avengers"),
            (
                "Lives of Others, The (Das leben der Anderen) (2006)",
                "The Lives of Others (Das leben der Anderen) (2006)",
            ),
            ("", ""),
            (None, ""),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(fix_title_display(raw), expected)


class TestAmbiguityHandling(unittest.TestCase):
    def test_avengers_ambiguous(self):
        """'Avengers' should return status 'ambiguous', structured matches with natural titles, and clarification message."""
        res = get_recommendations("Avengers")
        self.assertEqual(res.get("status"), "ambiguous")
        self.assertEqual(res.get("query"), "Avengers")
        self.assertEqual(res.get("recommendations"), [])
        self.assertIn("matches", res)
        self.assertTrue(len(res["matches"]) >= 2)

        # Verify matched titles are naturally formatted with release years
        match_titles = [m["title"] for m in res["matches"]]
        self.assertIn("The Avengers (1998)", match_titles)
        self.assertIn("The Avengers (2012)", match_titles)
        self.assertNotIn("Avengers, The (1998)", match_titles)
        self.assertNotIn("Avengers, The (2012)", match_titles)

        match_years = [m["year"] for m in res["matches"]]
        self.assertIn(1998, match_years)
        self.assertIn(2012, match_years)

        # Verify dynamic message includes the query and an example
        self.assertIn("Avengers", res["message"])
        self.assertIn("The Avengers (2012)", res["message"])
        self.assertTrue(res["message"].startswith("Multiple movies matched"))

    def test_avengers_query_with_title_parameter(self):
        """Endpoint should accept 'title' parameter identically to 'movie' parameter."""
        res = get_recommendations(title="Avengers")
        self.assertEqual(res.get("status"), "ambiguous")
        self.assertEqual(res.get("query"), "Avengers")
        match_titles = [m["title"] for m in res["matches"]]
        self.assertIn("The Avengers (1998)", match_titles)
        self.assertIn("The Avengers (2012)", match_titles)

    def test_batman_ambiguous(self):
        """'Batman' should return status 'ambiguous', structured matches, and a dynamic clarification message."""
        res = get_recommendations("Batman")
        self.assertEqual(res.get("status"), "ambiguous")
        self.assertEqual(res.get("query"), "Batman")
        self.assertEqual(res.get("recommendations"), [])
        self.assertIn("matches", res)
        self.assertTrue(len(res["matches"]) >= 2)

        match_titles = [m["title"] for m in res["matches"]]
        self.assertIn("Batman (1989)", match_titles)
        self.assertIn("Batman (1966)", match_titles)

        match_years = [m["year"] for m in res["matches"]]
        self.assertIn(1989, match_years)
        self.assertIn(1966, match_years)

        self.assertIn("Batman", res["message"])
        self.assertIn("Batman (1989)", res["message"])
        self.assertTrue(res["message"].startswith("Multiple movies matched"))

    def test_ambiguity_suggestion_chips_resolve_successfully(self):
        """Clicking any ambiguity suggestion chip title must resolve directly to recommendations."""
        res = get_recommendations("Avengers")
        self.assertEqual(res.get("status"), "ambiguous")
        matches = res.get("matches", [])
        self.assertTrue(len(matches) >= 2)

        for match in matches:
            suggested_title = match["title"]
            rec_res = get_recommendations(suggested_title)
            self.assertNotEqual(
                rec_res.get("status"),
                "ambiguous",
                f"Suggested chip title '{suggested_title}' should not trigger ambiguity again",
            )
            self.assertIn("recommendations", rec_res)
            self.assertEqual(
                len(rec_res["recommendations"]),
                10,
                f"Suggested chip title '{suggested_title}' should return 10 recommendations",
            )

    def test_year_qualified_query_resolves_normally(self):
        """A year-qualified query such as 'The Avengers (2012)' must return normal recommendations."""
        res = get_recommendations("The Avengers (2012)")
        self.assertNotEqual(res.get("status"), "ambiguous")
        self.assertIn("recommendations", res)
        self.assertEqual(len(res["recommendations"]), 10)
        self.assertTrue(
            all("title" in r and "score" in r for r in res["recommendations"])
        )

    def test_unfound_query_returns_normal_no_results(self):
        """An unfound query must return empty recommendations and not be flagged as ambiguous."""
        res = get_recommendations("asdfghjklqwerty1234567890")
        self.assertNotEqual(res.get("status"), "ambiguous")
        self.assertEqual(res.get("recommendations"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
