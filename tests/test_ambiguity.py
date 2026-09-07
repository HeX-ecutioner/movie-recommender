import unittest
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.index import get_recommendations, movies
from src.engine import resolve_movie_title


class TestAmbiguityHandling(unittest.TestCase):
    def test_avengers_ambiguous(self):
        """'Avengers' should return status 'ambiguous', structured matches, and a dynamic clarification message."""
        res = get_recommendations("Avengers")
        self.assertEqual(res.get("status"), "ambiguous")
        self.assertEqual(res.get("query"), "Avengers")
        self.assertEqual(res.get("recommendations"), [])
        self.assertIn("matches", res)
        self.assertTrue(len(res["matches"]) >= 2)
        
        # Verify matched titles contain release years
        match_years = [m["year"] for m in res["matches"]]
        self.assertIn(1998, match_years)
        self.assertIn(2012, match_years)
        
        # Verify dynamic message includes the query and an example
        self.assertIn("Avengers", res["message"])
        self.assertIn("The Avengers (2012)", res["message"])
        self.assertTrue(res["message"].startswith("Multiple movies matched"))

    def test_batman_ambiguous(self):
        """'Batman' should return status 'ambiguous', structured matches, and a dynamic clarification message."""
        res = get_recommendations("Batman")
        self.assertEqual(res.get("status"), "ambiguous")
        self.assertEqual(res.get("query"), "Batman")
        self.assertEqual(res.get("recommendations"), [])
        self.assertIn("matches", res)
        self.assertTrue(len(res["matches"]) >= 2)
        
        # Verify matched titles contain release years (1989 and 1966)
        match_years = [m["year"] for m in res["matches"]]
        self.assertIn(1989, match_years)
        self.assertIn(1966, match_years)
        
        # Verify dynamic message includes the query and an example
        self.assertIn("Batman", res["message"])
        self.assertIn("Batman (1989)", res["message"])
        self.assertTrue(res["message"].startswith("Multiple movies matched"))

    def test_year_qualified_query_resolves_normally(self):
        """A year-qualified query such as 'The Avengers (2012)' must return normal recommendations."""
        res = get_recommendations("The Avengers (2012)")
        self.assertNotEqual(res.get("status"), "ambiguous")
        self.assertIn("recommendations", res)
        self.assertEqual(len(res["recommendations"]), 10)
        self.assertTrue(all("title" in r and "score" in r for r in res["recommendations"]))

    def test_unfound_query_returns_normal_no_results(self):
        """An unfound query must return empty recommendations and not be flagged as ambiguous."""
        res = get_recommendations("asdfghjklqwerty1234567890")
        self.assertNotEqual(res.get("status"), "ambiguous")
        self.assertEqual(res.get("recommendations"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
