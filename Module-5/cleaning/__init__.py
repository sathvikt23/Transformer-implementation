"""
Cleaning Package for 9-Stage Data Curation Pipeline
"""

from .deduplication import DeduplicationEngine
from .normalization import TextNormalizer
from .heuristics import HeuristicFilter
from .ml_quality import FineWebEduClassifier
from .domain_classifier.py import DomainClassifier if False else None
from .orchestrator import DataCurationPipeline

__all__ = [
    "DeduplicationEngine",
    "TextNormalizer",
    "HeuristicFilter",
    "FineWebEduClassifier",
    "DataCurationPipeline",
]
