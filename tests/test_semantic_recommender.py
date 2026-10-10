import unittest

from semantic_recommender import SemanticBookRecommender


class SemanticBookRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recommender = SemanticBookRecommender.from_csv("data/books.csv")

    def test_returns_relevant_result_for_space_query(self) -> None:
        results = self.recommender.recommend("astronaut stranded on mars", top_k=1)
        self.assertEqual(results[0].title, "The Martian")

    def test_filters_by_category(self) -> None:
        results = self.recommender.recommend("habits productivity", top_k=5, category="Nonfiction")
        self.assertTrue(results)
        self.assertTrue(all(result.category == "Nonfiction" for result in results))

    def test_filters_by_emotion(self) -> None:
        results = self.recommender.recommend("science and survival", top_k=5, emotion="Hopeful")
        self.assertTrue(results)
        self.assertTrue(all(result.emotion == "Hopeful" for result in results))


if __name__ == "__main__":
    unittest.main()
