import pytest

from param_decomp.infra.wandb import fetch_latest_checkpoint_name


@pytest.mark.parametrize(
    "filenames, expected",
    [
        (["model_10.pth", "model_2.pth", "model_100.pth"], "model_100.pth"),
        (["model_2.pth", "model_10.pth"], "model_10.pth"),  # numeric, not lexicographic
        # The docstring's second supported format among numbered files used to
        # raise ValueError: int("model").
        (["model.pth", "model_100.pth"], "model_100.pth"),
        (["model.pth", "model_5.pth", "model_50.pth"], "model_50.pth"),
        # `.pt` files come through the W&B caller's (".pth", ".pt") filter;
        # int("100.pt") used to raise.
        (["model_10.pt", "model_100.pt"], "model_100.pt"),
        (["model_100.pth", "model_200.pt"], "model_200.pt"),
        # Word suffixes are stepless, never a crash.
        (["model_final.pth", "model_7.pth"], "model_7.pth"),
        # All stepless: deterministic lexicographic resolution.
        (["beta.pth", "alpha.pth"], "beta.pth"),
        # A lone file is returned as-is whatever it looks like.
        (["model.pth"], "model.pth"),
        (["model_final.pth"], "model_final.pth"),
    ],
    ids=[
        "max-step",
        "numeric-not-lexicographic",
        "bare-name-among-numbered",
        "bare-name-lowest-rank",
        "pt-suffixes",
        "mixed-pth-pt",
        "word-suffix-stepless",
        "all-stepless-lexicographic",
        "single-bare",
        "single-word-suffix",
    ],
)
def test_fetch_latest_checkpoint_name(filenames: list[str], expected: str) -> None:
    assert fetch_latest_checkpoint_name(filenames) == expected


def test_prefix_filters_before_ranking() -> None:
    filenames = ["other_999.pth", "model_10.pth", "model_100.pth"]
    assert fetch_latest_checkpoint_name(filenames, prefix="model_") == "model_100.pth"


def test_no_matches_raises() -> None:
    with pytest.raises(ValueError, match="No files found with prefix"):
        fetch_latest_checkpoint_name(["model_1.pth"], prefix="absent_")
