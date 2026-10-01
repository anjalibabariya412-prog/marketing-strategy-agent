import re
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

from backend.app.models.parsed_url import ParsedURL


# Regex for basic domain/hostname sanity check
DOMAIN_PATTERN = re.compile(
    r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
)


def parse_single_url(raw_url: str) -> ParsedURL:
    """
    Parses, validates, normalizes, and identifies the platform for a single URL string.
    Does not crash on invalid input; returns ParsedURL with is_valid=False on error.
    """
    stripped_url = raw_url.strip()
    if not stripped_url:
        return ParsedURL(
            original_url=raw_url,
            normalized_url=None,
            platform="unknown",
            is_valid=False,
            error="Empty URL string"
        )

    # Determine scheme. If missing, prepend https:// for parsing purposes.
    has_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', stripped_url))
    parse_target = stripped_url if has_scheme else f"https://{stripped_url}"

    try:
        parsed = urlparse(parse_target)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname.lower() if parsed.hostname else ""

        # Require HTTP or HTTPS scheme
        if scheme not in ("http", "https"):
            return ParsedURL(
                original_url=raw_url,
                normalized_url=None,
                platform="unknown",
                is_valid=False,
                error=f"Unsupported URL scheme '{scheme}'. Only HTTP/HTTPS are supported."
            )

        # Require a valid hostname
        if not hostname:
            return ParsedURL(
                original_url=raw_url,
                normalized_url=None,
                platform="unknown",
                is_valid=False,
                error="Invalid URL: missing hostname"
            )

        # Validate hostname domain format
        if not DOMAIN_PATTERN.match(hostname) and hostname != "localhost":
            return ParsedURL(
                original_url=raw_url,
                normalized_url=None,
                platform="unknown",
                is_valid=False,
                error="Invalid domain or hostname format"
            )

        # Build normalized URL
        netloc = hostname
        if parsed.port and not (
            (scheme == "http" and parsed.port == 80) or
            (scheme == "https" and parsed.port == 443)
        ):
            netloc = f"{hostname}:{parsed.port}"

        normalized_url = urlunparse((
            scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

        # Detect platform based on clean domain name
        clean_domain = hostname
        if clean_domain.startswith("www."):
            clean_domain = clean_domain[4:]

        if clean_domain in ("instagram.com", "instagr.am") or clean_domain.endswith(".instagram.com"):
            platform = "instagram"
        elif clean_domain in ("facebook.com", "fb.com", "fb.watch") or clean_domain.endswith(".facebook.com"):
            platform = "facebook"
        elif clean_domain == "linkedin.com" or clean_domain.endswith(".linkedin.com"):
            platform = "linkedin"
        else:
            platform = "website"

        return ParsedURL(
            original_url=raw_url,
            normalized_url=normalized_url,
            platform=platform,
            is_valid=True,
            error=None
        )

    except Exception as e:
        return ParsedURL(
            original_url=raw_url,
            normalized_url=None,
            platform="unknown",
            is_valid=False,
            error=f"URL parsing failed: {str(e)}"
        )


def parse_website_social_links(text: Optional[str]) -> List[ParsedURL]:
    """
    Takes a multi-line string of URLs, splits by line, ignores empty lines,
    strips whitespace, and parses each URL into a ParsedURL object.
    """
    if not text:
        return []

    results: List[ParsedURL] = []
    lines = text.splitlines()

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        results.append(parse_single_url(cleaned))

    return results
