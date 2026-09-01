from typing import Dict, Any

try:
    from transformers import pipeline
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False


class DomainClassifier:
    """
    Stage 8: Domain Classification & Pool Tagging.
    Categorizes clean documents into pools D1-D4 using Zero-Shot Classification (BART-Large-MNLI).
    """

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer and HAS_PIPELINE
        self.classifier = None

        if self.use_transformer:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            except Exception as e:
                print(f"[Warning] Could not load BART classifier: {e}. Using fast rule-based domain tagger.")
                self.use_transformer = False

        self.labels = [
            "D1 Science (arXiv preprints, research papers)",
            "D1 Knowledge (Wikipedia, reference, educational text)",
            "D2 Code and Mathematics (source code, algorithms, math problems)",
            "D2 Literature (books, formal documentation, structured text)",
            "D3 General Web (diverse filtered web content, news, articles)",
            "D4 Community (forums, QA discussions, social commentary)",
        ]

    def classify_domain(self, text: str) -> Dict[str, Any]:
        """Classifies document into target pool D1-D4."""
        # Truncate text to 1000 characters for classification efficiency
        sample_text = text[:1000]

        if self.use_transformer and self.classifier:
            try:
                result = self.classifier(sample_text, candidate_labels=self.labels)
                top_label = result["labels"][0]
                top_score = result["scores"][0]
                pool = self._label_to_pool(top_label)
                return {"domain": pool, "label": top_label, "confidence": top_score}
            except Exception:
                pass

        # Fast Rule-Based Heuristic Tagger for local execution
        sample_lower = sample_text.lower()

        if any(kw in sample_lower for kw in ["def ", "class ", "import ", "function", "return", "int ", "const ", "math", "equation", "proof", "theorem"]):
            return {"domain": "D2 Code and Mathematics", "confidence": 0.85}
        elif any(kw in sample_lower for kw in ["abstract", "doi:", "arxiv", "methodology", "experiment", "hypothesis", "analysis", "dataset"]):
            return {"domain": "D1 Science", "confidence": 0.80}
        elif any(kw in sample_lower for kw in ["wikipedia", "referred to as", "established in", "born", "century", "history of"]):
            return {"domain": "D1 Knowledge", "confidence": 0.82}
        elif any(kw in sample_lower for kw in ["chapter", "novel", "manual", "guide", "documentation", "section"]):
            return {"domain": "D2 Literature", "confidence": 0.75}
        elif any(kw in sample_lower for kw in ["forum", "reddit", "posted by", "comment", "thread", "reply", "user"]):
            return {"domain": "D4 Community", "confidence": 0.78}
        else:
            return {"domain": "D3 General Web", "confidence": 0.70}

    def _label_to_pool(self, label: str) -> str:
        if "Science" in label:
            return "D1 Science"
        elif "Knowledge" in label:
            return "D1 Knowledge"
        elif "Code" in label:
            return "D2 Code and Mathematics"
        elif "Literature" in label:
            return "D2 Literature"
        elif "Community" in label:
            return "D4 Community"
        else:
            return "D3 General Web"
