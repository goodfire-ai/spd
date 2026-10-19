"""Reusable capture grammar for targets whose points are linear-site inputs and outputs.

TMS and ResidMLP deliberately share this small vocabulary. Each canonical chain value has
one public name; ``{site}.out`` names a site's distinct linear output. Transformer targets
do not use this helper; their richer point algebra lives in ``transformer_taps.py``. Both
modules stay target-side, so core owns neither syntax nor architecture anatomy.
"""

from collections.abc import Callable, Hashable
from dataclasses import dataclass

_OUTPUT_SUFFIX = ".out"


def site_output_key(site: str) -> str:
    return f"{site}{_OUTPUT_SUFFIX}"


@dataclass(frozen=True, kw_only=True)
class SiteInputPoint:
    site: str


@dataclass(frozen=True, kw_only=True)
class SiteOutputPoint:
    site: str


SitePoint = SiteInputPoint | SiteOutputPoint


SiteCaptureSources = tuple[SitePoint, ...]


@dataclass(frozen=True, kw_only=True)
class SiteCaptureGrammar:
    sites: tuple[str, ...]
    physical_source_of: Callable[[SitePoint], Hashable]

    def __post_init__(self) -> None:
        assert len(set(self.sites)) == len(self.sites), f"duplicate target sites {self.sites}"
        assert all(not site.endswith(_OUTPUT_SUFFIX) for site in self.sites), self.sites

    def parse(self, key: str) -> SitePoint:
        if key.endswith(_OUTPUT_SUFFIX):
            site = key.removesuffix(_OUTPUT_SUFFIX)
            assert site in self.sites, f"unknown site-output activation {key!r}"
            return SiteOutputPoint(site=site)
        assert key in self.sites, f"unknown site-input activation {key!r}"
        return SiteInputPoint(site=key)

    def resolve(self, keys: tuple[str, ...]) -> SiteCaptureSources:
        """Validate ``keys`` and return their typed points in request order."""
        points = tuple(self.parse(key) for key in keys)
        physical_sources = tuple(self.physical_source_of(point) for point in points)
        assert len(set(physical_sources)) == len(physical_sources), (
            "multiple capture keys name one physical activation",
            keys,
            physical_sources,
        )
        return points


@dataclass(frozen=True, kw_only=True)
class CaptureKeysBySite:
    input_key_by_site: dict[str, str]
    output_key_by_site: dict[str, str]


def capture_keys_by_site(keys: tuple[str, ...], sources: SiteCaptureSources) -> CaptureKeysBySite:
    """Index requested input and output keys by site."""
    input_key_by_site: dict[str, str] = {}
    output_key_by_site: dict[str, str] = {}
    for key, point in zip(keys, sources, strict=True):
        match point:
            case SiteInputPoint(site=site):
                assert site not in input_key_by_site, site
                input_key_by_site[site] = key
            case SiteOutputPoint(site=site):
                assert site not in output_key_by_site, site
                output_key_by_site[site] = key
    return CaptureKeysBySite(
        input_key_by_site=input_key_by_site, output_key_by_site=output_key_by_site
    )


def record_requested_value[T](captures: dict[str, T], key: str | None, value: T) -> None:
    """Record ``value`` only when the capture request includes this activation."""
    if key is not None:
        captures[key] = value
