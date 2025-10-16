from pathlib import Path

from spd.log import logger


def write_html_files(output_dir: Path) -> None:
    """Write bundled HTML files from _bundled to output directory.

    Args:
        output_dir: Directory to write HTML files to
    """
    import importlib.resources

    # Read bundled HTML files from the _bundled package
    bundled_package = "spd.clustering.dashboard._bundled"

    index_html = importlib.resources.files(bundled_package).joinpath("index.html").read_text()
    cluster_html = importlib.resources.files(bundled_package).joinpath("cluster.html").read_text()

    # Write to output directory
    (output_dir / "index.html").write_text(index_html)
    (output_dir / "cluster.html").write_text(cluster_html)

    logger.info(f"HTML files written to: {output_dir}")
