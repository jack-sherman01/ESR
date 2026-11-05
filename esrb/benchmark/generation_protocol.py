from dataclasses import dataclass


@dataclass
class GenerationProtocol:
    num_generations: int        # K
    rollouts_per_generation: int
    seed: int = 0
    reset_between_generations: bool = True
