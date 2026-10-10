import argparse
import json
from dataclasses import asdict

from semantic_recommender import SemanticBookRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic book recommender")
    parser.add_argument("--data", default="data/books.csv", help="Path to books CSV")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--category", help="Optional category filter")
    parser.add_argument("--emotion", help="Optional emotion filter")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommender = SemanticBookRecommender.from_csv(args.data)
    results = recommender.recommend(
        query=args.query,
        top_k=args.top_k,
        category=args.category,
        emotion=args.emotion,
        min_score=args.min_score,
    )
    print(json.dumps([asdict(item) for item in results], indent=2))


if __name__ == "__main__":
    main()
