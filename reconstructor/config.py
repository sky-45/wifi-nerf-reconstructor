"""
Centralized Configuration & Optimal Hyperparameters for Wi-Fi NeRF Floorplan Reconstructor.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os


@dataclass
class ReconstructorConfig:
    # Stage 1: Pose Graph SLAM & Free Space Carver
    free_space_radius_m: float = 0.35
    keyframe_dt_s: float = 1.0

    # Stage 3A: Fast FISTA + TV Sparse Tomography
    fista_iterations: int = 40
    fista_step_size: float = 0.03
    fista_l1_reg: float = 0.001
    fista_tv_reg: float = 0.01

    # Stage 3B: NeRFMap Coordinate Field Refinement
    fourier_bands: int = 8
    mlp_hidden_dim: int = 32
    nerf_epochs: int = 80
    nerf_lr: float = 0.01
    use_cuda: bool = True
    model_type: str = "fourier" # Options: 'fourier', 'hashgrid', 'siren', 'nerf_3d'
    siren_omega: float = 30.0
    hash_levels: int = 8
    learn_antenna_pattern: bool = False

    # Stage 4: CAD Vectorization & Room Extraction
    wall_threshold_percentile: float = 60
    resolution_m: float = 0.2
    min_wall_length_m: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructorConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# Default global hyperparameters instance
DEFAULT_CONFIG = ReconstructorConfig()


def get_default_config() -> ReconstructorConfig:
    """Returns a fresh instance of the default configuration."""
    return ReconstructorConfig.from_dict(DEFAULT_CONFIG.to_dict())


def save_optimal_config_to_py(best_params: Dict[str, Any], filepath: Optional[str] = None) -> None:
    from reconstructor.config import DEFAULT_CONFIG
    target_path = filepath or os.path.abspath(__file__)
    current = DEFAULT_CONFIG.to_dict()
    current.update(best_params)
    # Recursively write identical template structure
    from reconstructor.tune_hyperparameters import save_config_file_helper
    save_config_file_helper(current, target_path)
