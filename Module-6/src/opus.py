from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class OPUSDecision:
    candidate_id: str
    status: str  # accepted / rejected / deferred / protected-floor-rescued
    reason: str  # quality_above_threshold / quota_pressure / deferred_for_later_stage / protected_floor_rescued
    score: float
    lane: str
    rescued_by_protected_floor: bool


class OPUSEngine:
    """
    OPUS Candidate Scoring & Selection Engine.
    Filters candidate samples based on quality threshold, lane quota pressure,
    and protected floor guarantees.
    """

    def __init__(self, quality_threshold: float = 0.70):
        self.quality_threshold = quality_threshold
        # Protected floor lanes ensure critical underrepresented data (e.g., Indic) is never dropped completely
        self.protected_lanes = {"indic"}
        self.decisions_log: list = []

    def evaluate_candidate(
        self,
        candidate_id: str,
        score: float,
        lane: str,
        lane_quota_full: bool = False,
        defer: bool = False,
    ) -> OPUSDecision:
        """
        Evaluates a candidate sample and produces a deterministic OPUS decision.
        """
        rescued = False

        if defer:
            status = "deferred"
            reason = "deferred_for_later_stage"
        elif lane_quota_full:
            if lane in self.protected_lanes and score >= 0.50:
                status = "protected-floor-rescued"
                reason = "protected_floor_rescued"
                rescued = True
            else:
                status = "rejected"
                reason = "quota_pressure"
        elif score < self.quality_threshold:
            if lane in self.protected_lanes and score >= 0.50:
                status = "protected-floor-rescued"
                reason = "protected_floor_rescued"
                rescued = True
            else:
                status = "rejected"
                reason = "low_quality_score"
        else:
            status = "accepted"
            reason = "quality_above_threshold"

        decision = OPUSDecision(
            candidate_id=candidate_id,
            status=status,
            reason=reason,
            score=score,
            lane=lane,
            rescued_by_protected_floor=rescued,
        )

        self.decisions_log.append(decision)
        return decision

    def export_to_jsonl(self, filepath: str):
        import json
        import os
        from dataclasses import asdict, is_dataclass
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for d in self.decisions_log:
                if is_dataclass(d):
                    f.write(json.dumps(asdict(d)) + "\n")
                elif hasattr(d, "__dict__"):
                    f.write(json.dumps(d.__dict__) + "\n")
                elif isinstance(d, dict):
                    f.write(json.dumps(d) + "\n")

