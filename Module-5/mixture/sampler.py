import random
from typing import Dict, List, Any


class CurriculumSampler:
    """
    Curriculum Data Mixture Sampler for Pretraining Stages:
    - Seed / Early Stage (0-10%): High general web, low code/reasoning, floors on Indic (8%) & Agentic (3%).
    - Mid Stage (10-70%): Rising code & reasoning shares, expanded Indic & Agentic.
    - Late Stage (70-98%): High code & reasoning focus, full target floors.
    - Anneal Stage (98-100%): Concentrated deployment of verified Indic reserve, top agentic trajectories, and high/ultra reasoning.
    """

    def __init__(self, curriculum_stages: Dict[str, Dict[str, float]], total_token_budget: int = 100_000_000_000):
        self.stages = curriculum_stages
        self.total_budget = total_token_budget

    def get_stage_ratios(self, stage_name: str) -> Dict[str, float]:
        """Returns normalized target lane shares for a given curriculum stage."""
        stage = stage_name.lower()
        if stage not in self.stages:
            raise ValueError(f"Unknown curriculum stage '{stage_name}'. Valid stages: {list(self.stages.keys())}")

        ratios = self.stages[stage]
        total = sum(ratios.values())
        return {lane: round(share / total, 4) for lane, share in ratios.items()}

    def compute_token_allocations(self, stage_name: str, batch_token_count: int) -> Dict[str, int]:
        """Computes exact token allocations per lane for a batch in the specified stage."""
        ratios = self.get_stage_ratios(stage_name)
        allocations = {}
        allocated_so_far = 0

        lanes = list(ratios.keys())
        for lane in lanes[:-1]:
            tokens = int(batch_token_count * ratios[lane])
            allocations[lane] = tokens
            allocated_so_far += tokens

        # Assign remaining tokens to last lane to ensure exact batch sum
        allocations[lanes[-1]] = batch_token_count - allocated_so_far
        return allocations

    def sample_batch_sources(self, stage_name: str, available_pools: Dict[str, List[Any]], batch_size: int = 100) -> List[Any]:
        """Samples a training batch across domain pools according to current curriculum stage ratios."""
        ratios = self.get_stage_ratios(stage_name)
        batch = []

        for lane, share in ratios.items():
            pool = available_pools.get(lane, [])
            if not pool:
                continue

            count = max(1, int(batch_size * share))
            sampled = random.choices(pool, k=min(count, len(pool)))
            batch.extend(sampled)

        random.shuffle(batch)
        return batch[:batch_size]
