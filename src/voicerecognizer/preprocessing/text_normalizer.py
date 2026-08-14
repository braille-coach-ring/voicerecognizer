import unicodedata


class TextNormalizer:
    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text).strip()
