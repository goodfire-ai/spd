"""Search for strings in the SimpleStories dataset."""

import os
import re
from pathlib import Path
from typing import Any

import fire
from datasets import Dataset, load_dataset

from spd.log import logger
from spd.settings import REPO_ROOT


def _highlight_matches(text: str, query: str) -> str:
    """Wrap all occurrences of query in text with markdown bold markers."""
    pattern = re.escape(query)
    return re.sub(pattern, r"**\g<0>**", text, flags=re.IGNORECASE)


def search(
    query: str,
    output_dir: str | None = None,
    split: str = "train",
    n_write_stories: int | None = 1000,
) -> None:
    """Search for a string in the SimpleStories dataset (case-insensitive).

    Args:
        query: The string to search for
        output_dir: Directory for output markdown file (default: outputs/dataset_search)
        split: Dataset split to search ('train' or 'test')
        n_write_stories: Maximum number of matching stories to write to file (None = all)

    Examples:
        dataset-search "the cat"                    # Search for "the cat"
        dataset-search "dog" --split test           # Search test split
        dataset-search "magic" --n_write_stories 10 # Limit output to 10 stories
    """
    assert query, "Query string cannot be empty"
    assert split in ("train", "test"), f"Split must be 'train' or 'test', got '{split}'"

    out_path = REPO_ROOT / "out" / "dataset_search" if output_dir is None else Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SimpleStories dataset (split={split})...")
    dataset = load_dataset("SimpleStories/SimpleStories", split=split)
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    logger.info(f"Searching for '{query}'...")

    search_query = query.lower()

    def contains_query(example: dict[str, Any]) -> bool:
        story: str = example["story"]
        return search_query in story.lower()

    num_proc = min(os.cpu_count() or 1, 8)
    logger.info(f"Filtering with {num_proc} processes...")
    filtered_dataset = dataset.filter(contains_query, num_proc=num_proc)

    matching_stories: list[dict[str, Any]] = []
    total_occurrences = 0

    for item in filtered_dataset:
        story: str = item["story"]  # pyright: ignore[reportArgumentType,reportCallIssue]
        count = story.lower().count(search_query)
        total_occurrences += count
        matching_stories.append(
            {
                "story": story,
                "occurrences": count,
                "topic": item["topic"],  # pyright: ignore[reportArgumentType,reportCallIssue]
                "theme": item["theme"],  # pyright: ignore[reportArgumentType,reportCallIssue]
            }
        )

    total_stories = len(dataset)
    logger.info(
        f"Found {total_occurrences} occurrences in {len(matching_stories)} stories "
        f"(searched {total_stories} stories)"
    )

    safe_query = "".join(c if c.isalnum() or c in " -_" else "_" for c in query)
    safe_query = safe_query.replace(" ", "_")[:50]
    output_file = out_path / f"search_{safe_query}.md"

    stories_to_write = matching_stories
    if n_write_stories is not None and len(matching_stories) > n_write_stories:
        stories_to_write = matching_stories[:n_write_stories]
        logger.info(
            f"Limiting output to {n_write_stories} stories (of {len(matching_stories)} total)"
        )

    md_content = _generate_markdown(
        query=query,
        split=split,
        total_stories_searched=total_stories,
        total_occurrences=total_occurrences,
        total_matching_stories=len(matching_stories),
        stories=stories_to_write,
    )

    output_file.write_text(md_content)
    logger.info(f"Results saved to: {output_file}")


def _generate_markdown(
    query: str,
    split: str,
    total_stories_searched: int,
    total_occurrences: int,
    total_matching_stories: int,
    stories: list[dict[str, Any]],
) -> str:
    """Generate markdown content for search results."""
    lines = [
        "# Dataset Search Results",
        "",
        f"**Query:** `{query}`",
        f"**Split:** {split}",
        "",
        "## Summary",
        "",
        f"- **Total Stories Searched:** {total_stories_searched}",
        f"- **Total Occurrences:** {total_occurrences}",
        f"- **Stories Containing Query:** {total_matching_stories}",
        f"- **Stories Shown:** {len(stories)}",
        "",
        "---",
        "",
        "## Matching Stories",
        "",
    ]

    for i, item in enumerate(stories, 1):
        lines.extend(
            [
                f"### Story {i}",
                "",
                f"**Occurrences in this story:** {item['occurrences']}",
            ]
        )
        if item["topic"]:
            lines.append(f"**Topic:** {item['topic']}")
        if item["theme"]:
            lines.append(f"**Theme:** {item['theme']}")

        highlighted_story = _highlight_matches(item["story"], query)
        lines.extend(
            [
                "",
                highlighted_story,
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines)


def cli() -> None:
    fire.Fire(search)


if __name__ == "__main__":
    cli()
