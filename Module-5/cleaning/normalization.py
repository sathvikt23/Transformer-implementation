import re
import unicodedata
from bs4 import BeautifulSoup

try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False


class TextNormalizer:
    """
    Stage 3: Text Normalization & Repair
    Stage 7: Privacy Protection (PII Scrubbing)
    """

    def __init__(self):
        # Email & URL Regex patterns for inline PII anonymization
        self.email_regex = re.compile(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.IGNORECASE
        )
        self.url_regex = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            re.IGNORECASE,
        )

    def normalize_text(self, text: str) -> str:
        """Stage 3: Strips HTML, repairs corrupted encodings, normalizes Unicode NFKC."""
        if not text:
            return ""

        # 1. Strip raw HTML tags if present
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ")

        # 2. Fix text encodings via ftfy
        if HAS_FTFY:
            text = ftfy.fix_text(text)

        # 3. NFKC Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # 4. Collapse extra whitespaces & control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def redact_pii(self, text: str) -> str:
        """Stage 7: Inline scrubbing of PII (emails and URLs) without dropping documents."""
        text = self.email_regex.sub("[REDACTED_EMAIL]", text)
        text = self.url_regex.sub("[REDACTED_URL]", text)
        return text
