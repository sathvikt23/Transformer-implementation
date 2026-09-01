from typing import Dict, Any, Tuple, Optional
from .deduplication import DeduplicationEngine
from .normalization import TextNormalizer
from .heuristics import HeuristicFilter
from .ml_quality import FineWebEduClassifier
from .domain_classifier import DomainClassifier


class DataCurationPipeline:
    """
    Complete 9-Stage Data Curation Pipeline Funnel.
    Applies the full sequence of cleaning, repair, filtering, and tagging rules.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dedup = DeduplicationEngine(
            num_perm=config.get("dedup", {}).get("num_perm", 128),
            jaccard_threshold=config.get("dedup", {}).get("jaccard_threshold", 0.90)
        )
        self.normalizer = TextNormalizer()
        self.heuristics = HeuristicFilter(config.get("filters", {}))
        self.ml_classifier = FineWebEduClassifier(
            threshold=config.get("fineweb_threshold", 0.50)
        )
        self.domain_classifier = DomainClassifier()

        # Metrics Tracking Funnel Counters
        self.stats = {
            "total_input": 0,
            "stage1_lang_drop": 0,
            "stage2_exact_dedup_drop": 0,
            "stage2_near_dedup_drop": 0,
            "stage3_normalized": 0,
            "stage4_heuristic_drop": 0,
            "stage5_ml_quality_drop": 0,
            "stage6_task_filter_drop": 0,
            "stage7_pii_redacted": 0,
            "stage8_retained": 0,
        }

    def process_document(self, raw_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Executes raw document through the 9-Stage Pipeline.
        Returns (cleaned_text, metadata) if document passes all filters, else (None, None).
        """
        self.stats["total_input"] += 1

        # Stage 1: Language Filtering (Detect English / target)
        if not self.heuristics.is_english(raw_text):
            self.stats["stage1_lang_drop"] += 1
            return None, None

        # Stage 2: Exact Deduplication
        if self.dedup.is_exact_duplicate(raw_text):
            self.stats["stage2_exact_dedup_drop"] += 1
            return None, None

        # Stage 2: Near Deduplication (MinHash LSH)
        if self.dedup.is_near_duplicate(raw_text):
            self.stats["stage2_near_dedup_drop"] += 1
            return None, None

        # Stage 3: Text Normalization & Repair
        text = self.normalizer.normalize_text(raw_text)
        self.stats["stage3_normalized"] += 1

        # Stage 4: Heuristic Quality Filtering
        if not self.heuristics.passes_heuristic_quality(text):
            self.stats["stage4_heuristic_drop"] += 1
            return None, None

        # Stage 5: ML Quality Scoring (FineWeb-Edu)
        if not self.ml_classifier.passes_quality(text):
            self.stats["stage5_ml_quality_drop"] += 1
            return None, None

        # Stage 6: Custom Task Filtering (Syntax & Density)
        if not self.heuristics.passes_custom_task_filter(text):
            self.stats["stage6_task_filter_drop"] += 1
            return None, None

        # Stage 7: Privacy Protection (PII Scrubbing)
        text = self.normalizer.redact_pii(text)
        self.stats["stage7_pii_redacted"] += 1

        # Stage 8: Domain Tagging (BART-Large-MNLI)
        domain_meta = self.domain_classifier.classify_domain(text)
        self.stats["stage8_retained"] += 1

        metadata = {
            "domain": domain_meta["domain"],
            "confidence": domain_meta["confidence"],
            "word_count": len(text.split()),
            "char_count": len(text),
        }

        return text, metadata

    def get_funnel_summary(self) -> Dict[str, Any]:
        """Returns statistical summary of filter funnel yield."""
        retained = self.stats["stage8_retained"]
        total = max(1, self.stats["total_input"])
        yield_pct = (retained / total) * 100.0
        return {
            "total_input_docs": total,
            "retained_docs": retained,
            "yield_percentage": round(yield_pct, 2),
            "noise_filtered_pct": round(100.0 - yield_pct, 2),
            "drops": {
                "language_filter": self.stats["stage1_lang_drop"],
                "exact_dedup": self.stats["stage2_exact_dedup_drop"],
                "near_dedup_minhash": self.stats["stage2_near_dedup_drop"],
                "heuristic_quality": self.stats["stage4_heuristic_drop"],
                "fineweb_ml_quality": self.stats["stage5_ml_quality_drop"],
                "syntax_task_filter": self.stats["stage6_task_filter_drop"],
            },
            "pii_redacted_docs": self.stats["stage7_pii_redacted"],
        }
