import math
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class FineWebEduClassifier:
    """
    Stage 5: ML-Based Educational Quality Scoring.
    Evaluates documents using HuggingFaceTB/FineWeb-Edu-classifier or a high-precision fallback.
    Documents with score >= threshold (default 0.50) are retained.
    """

    def __init__(self, model_name: str = "HuggingFaceTB/FineWeb-Edu-classifier", threshold: float = 0.50, use_ml_model: bool = False):
        self.threshold = threshold
        self.use_ml_model = use_ml_model and HAS_TRANSFORMERS
        self.tokenizer = None
        self.model = None

        if self.use_ml_model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
            except Exception as e:
                print(f"[Warning] Could not load FineWeb-Edu model '{model_name}': {e}. Using rule-based ML quality estimator.")
                self.use_ml_model = False

    def predict_score(self, text: str) -> float:
        """Computes educational quality score between 0.0 and 5.0 (or normalized 0.0 to 1.0)."""
        if self.use_ml_model and self.model and self.tokenizer:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                score = torch.sigmoid(logits).item()
                return score

        # Fast CPU Heuristic Estimator (mimicking FineWeb-Edu criteria: structure, depth, educational markers)
        words = text.split()
        if len(words) < 20:
            return 0.10

        # Educational marker keywords
        edu_keywords = {
            "because", "therefore", "however", "example", "result", "analysis",
            "study", "research", "method", "definition", "figure", "table", "section",
            "theory", "function", "system", "algorithm", "data", "process", "model"
        }
        word_set = set(w.lower() for w in words)
        edu_matches = len(word_set.intersection(edu_keywords))

        # Average word length & vocabulary complexity score
        avg_word_len = sum(len(w) for w in words) / len(words)
        complexity_score = min(1.0, max(0.0, (avg_word_len - 3.5) / 3.0))

        # Normalized educational density score
        score = 0.30 + (0.40 * min(1.0, edu_matches / 5.0)) + (0.30 * complexity_score)
        return min(1.0, score)

    def passes_quality(self, text: str) -> bool:
        """Returns True if document score meets or exceeds FineWeb-Edu threshold."""
        score = self.predict_score(text)
        return score >= self.threshold
