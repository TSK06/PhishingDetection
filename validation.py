from urllib.parse import urlsplit


MAX_URL_LENGTH = 2048


def validate_url(value: str) -> str:
    url = value.strip()
    if not url or len(url) > MAX_URL_LENGTH:
        raise ValueError(f"URL must be between 1 and {MAX_URL_LENGTH} characters.")
    if any(ord(character) < 32 for character in url):
        raise ValueError("URL contains unsupported control characters.")

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("Enter a valid HTTP or HTTPS URL.") from exc

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or any(character.isspace() for character in hostname)
    ):
        raise ValueError("Enter a valid HTTP or HTTPS URL.")
    return url
