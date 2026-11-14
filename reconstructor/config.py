"""
Centralized Configuration for 3D Camping & Multi-Floor Wi-Fi Environment Reconstructor.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ReconstructorConfig:
    # Stage 0: Camping Dwell Compression & Drift Guards
    camping_dwell_radius_m: float = 0.15
    camping_heading_tol_deg: float = 15.0
    camping_window_s: float = 10.0
    max_human_speed_mps: float = 3.5

    # Stage 1: Pose Graph SLAM & Free Space Carver
    free_space_radius_m: float = 0.45
    keyframe_dt_s: float = 1.0

    # Stage 3: Volumetric 3D Tomography & Shadow-Shell Inversion
    resolution_m: float = 0.25
    z_resolution_m: float = 0.50
    z_min_m: float = -4.0
    z_max_m: float = 6.0
    fista_iterations: int = 45
    fista_step_size: float = 0.35
    fista_l1_reg: float = 0.005
    fista_tv_reg: float = 0.02
    los_clear_excess_db: float = 1.5
    use_cuda: bool = True

    # Stage 4: Physical Wall Extraction & Threshold Gating
    wall_threshold_db_m: float = 1.8
    wall_threshold_percentile: float = 75.0
    min_wall_length_m: float = 0.70
    min_wall_ray_hits: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructorConfig":
        valid_keys = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


DEFAULT_CONFIG = ReconstructorConfig()


def get_default_config() -> ReconstructorConfig:
    """Returns a fresh instance of the default configuration."""
    return ReconstructorConfig.from_dict(DEFAULT_CONFIG.to_dict())
