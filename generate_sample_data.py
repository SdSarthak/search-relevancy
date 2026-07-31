"""Generate a synthetic news corpus so the pipeline can be run end to end.

No real articles are shipped with this repository. This script fabricates a
topically-structured corpus (each article is built from one topic's vocabulary)
which is enough to exercise preprocessing, embedding, indexing and the
relevancy evaluation.

    python generate_sample_data.py --num-articles 500
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# topic -> (category, subcategory, headline fragments, body sentences)
TOPICS: Dict[str, dict] = {
    "climate": {
        "category": "Science",
        "subcategory": "Environment",
        "headlines": [
            "ocean temperatures reach new high",
            "glacier retreat accelerates in the alps",
            "carbon emissions plateau across europe",
            "coral reefs show signs of recovery",
        ],
        "sentences": [
            "Researchers report that rising temperatures continue to reshape regional weather patterns.",
            "Long term monitoring stations recorded another year of above average sea surface temperatures.",
            "Emission reduction pledges remain short of the trajectory needed to limit warming.",
            "Ecologists warn that shifting rainfall is stressing freshwater ecosystems.",
        ],
    },
    "ai": {
        "category": "Technology",
        "subcategory": "AI",
        "headlines": [
            "language models shrink without losing accuracy",
            "chip shortage slows model training",
            "researchers publish new benchmark for reasoning",
            "retrieval systems close the gap on larger models",
        ],
        "sentences": [
            "The team reports competitive accuracy at a fraction of the parameter count.",
            "Training throughput improved after the scheduler was rewritten to overlap communication.",
            "Evaluation on held out tasks suggests the gains generalise beyond the benchmark.",
            "Engineers describe the deployment as a straightforward drop in replacement.",
        ],
    },
    "markets": {
        "category": "Business",
        "subcategory": "Markets",
        "headlines": [
            "equities rally on softer inflation print",
            "central bank holds rates steady",
            "quarterly earnings beat analyst expectations",
            "bond yields slip after auction",
        ],
        "sentences": [
            "Investors reacted to the release by rotating out of defensive sectors.",
            "The company raised guidance for the remainder of the financial year.",
            "Analysts noted that margins held up despite higher input costs.",
            "Trading volumes were thin ahead of the holiday shortened week.",
        ],
    },
    "health": {
        "category": "Science",
        "subcategory": "Medical",
        "headlines": [
            "trial reports fewer side effects",
            "hospitals adopt remote monitoring",
            "vaccine coverage improves in rural districts",
            "study links sleep quality to recovery",
        ],
        "sentences": [
            "The randomised trial enrolled several thousand participants across twelve sites.",
            "Clinicians say the protocol shortens the average length of stay.",
            "Public health officials attributed the improvement to mobile outreach teams.",
            "Follow up data will be published once the second cohort completes treatment.",
        ],
    },
    "politics": {
        "category": "World",
        "subcategory": "Politics",
        "headlines": [
            "coalition talks enter a second week",
            "delegates agree on trade framework",
            "parliament debates budget amendment",
            "diplomats resume stalled negotiations",
        ],
        "sentences": [
            "Negotiators described the discussions as constructive but incomplete.",
            "The proposal would phase in tariff reductions over the next four years.",
            "Opposition members requested an independent review of the spending plan.",
            "A joint statement is expected before the end of the summit.",
        ],
    },
    "sport": {
        "category": "Entertainment",
        "subcategory": "Events",
        "headlines": [
            "underdog side reaches the final",
            "record broken at the national championship",
            "stadium renovation completed ahead of schedule",
            "veteran captain announces retirement",
        ],
        "sentences": [
            "The result caps an unlikely run that began in the qualifying rounds.",
            "Officials confirmed the time after reviewing the timing system.",
            "Organisers expect capacity crowds for the opening fixture.",
            "Teammates praised the leadership shown throughout the campaign.",
        ],
    },
}

SOURCES = [
    "BBC",
    "Reuters",
    "CNN",
    "Associated Press",
    "The Guardian",
    "New York Times",
    "TechCrunch",
    "Nature",
    "Science Daily",
]


def generate_sample_data(num_articles: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic synthetic corpus of ``num_articles`` rows."""
    if num_articles <= 0:
        raise ValueError("num_articles must be positive")

    rng = random.Random(seed)
    topic_names = list(TOPICS)
    base_date = datetime(2024, 1, 1)

    articles: List[dict] = []
    for position in range(num_articles):
        topic_name = topic_names[position % len(topic_names)]
        topic = TOPICS[topic_name]
        headline = rng.choice(topic["headlines"])
        body = " ".join(rng.sample(topic["sentences"], k=3))

        articles.append(
            {
                "article_id": f"article_{position + 1:05d}",
                "category": topic["category"],
                "subcategory": topic["subcategory"],
                "title": f"{topic_name.title()}: {headline}",
                "published_date": (
                    base_date + timedelta(days=rng.randint(0, 364))
                ).strftime("%Y-%m-%d"),
                "text": f"{headline.capitalize()}. {body}",
                "source": rng.choice(SOURCES),
            }
        )
    return pd.DataFrame(articles)


def main(argv=None) -> int:
    default_output = Path(__file__).parent / "data" / "raw" / "news_articles.csv"

    parser = argparse.ArgumentParser(description="Generate a synthetic news corpus.")
    parser.add_argument("--num-articles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args(argv)

    df = generate_sample_data(args.num_articles, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"wrote {len(df)} synthetic articles to {output_path}")
    print(df[["article_id", "category", "title"]].head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
