from typing import Dict, List, Any, Optional
from src.shards import Shard


STAGE_PROPORTIONS = {
    "pretrain-base": {
        "code": 0.25,
        "reasoning": 0.25,
        "indic": 0.10,
        "general": 0.40,
    },
    "reasoning-heavy-midtrain": {
        "code": 0.30,
        "reasoning": 0.45,
        "indic": 0.10,
        "general": 0.15,
    },
    "code-boost-anneal": {
        "code": 0.50,
        "reasoning": 0.30,
        "indic": 0.10,
        "general": 0.10,
    },
}


class MixtureScheduler:
    """
    Curriculum Mixture Scheduler managing lane proportions and quotas per training stage.
    """

    def __init__(self, current_stage: str = "reasoning-heavy-midtrain"):
        if current_stage not in STAGE_PROPORTIONS:
            raise ValueError(f"Unknown curriculum stage: {current_stage}")
        self.current_stage = current_stage
        self.proportions = STAGE_PROPORTIONS[current_stage]
        self.served_counts: Dict[str, int] = {lane: 0 for lane in self.proportions}

    def set_stage(self, stage_name: str):
        if stage_name not in STAGE_PROPORTIONS:
            raise ValueError(f"Unknown curriculum stage: {stage_name}")
        self.current_stage = stage_name
        self.proportions = STAGE_PROPORTIONS[stage_name]

    def get_target_quota(self, total_batches: int) -> Dict[str, int]:
        """Calculates exact per-lane batch quota targets for a total batch budget."""
        return {lane: int(prop * total_batches) for lane, prop in self.proportions.items()}

    def record_served(self, lane: str):
        if lane in self.served_counts:
            self.served_counts[lane] += 1
        else:
            self.served_counts[lane] = 1

    def get_actual_shares(self) -> Dict[str, float]:
        total = sum(self.served_counts.values())
        if total == 0:
            return {lane: 0.0 for lane in self.proportions}
        return {lane: count / total for lane, count in self.served_counts.items()}

    def select_lane_for_step(self, step: int) -> str:
        """
        Determines lane for step using round-robin / proportion-matching schedule.
        """
        # Calculate ideal total shares vs current served counts
        lanes = list(self.proportions.keys())
        total_served = sum(self.served_counts.values())
        
        if total_served == 0:
            return lanes[0]

        max_deficit = -1.0
        selected_lane = lanes[0]

        for lane, target_prop in self.proportions.items():
            actual_prop = self.served_counts.get(lane, 0) / total_served
            deficit = target_prop - actual_prop
            if deficit > max_deficit:
                max_deficit = deficit
                selected_lane = lane

        return selected_lane
