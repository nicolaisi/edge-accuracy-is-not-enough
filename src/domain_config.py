"""
Domain configuration for multi-domain structure-transfer experiments.

Each domain dict contains all hard-coded constants that differ between
the CFD (hydrogen parking garage) domain and the acoustic domain.
"""

CFD_CONFIG = {
    "name": "cfd",
    "bounds": (50.0, 30.0, 3.0),          # metres
    "eval_x_range": (5.0, 45.0),
    "eval_y_range": (5.0, 25.0),
    "eval_z": 2.0,
    "detection_threshold": 50.0,           # ppm
    "ood_split_key": "ventilation",
    "ood_train_values": [6, 10],
    "ood_test_values": [3],
}

ACOUSTIC_CONFIG = {
    "name": "acoustic",
    "bounds": (20.0, 15.0, 0.1),           # metres (dummy z for 2D→3D)
    "eval_x_range": (1.0, 19.0),
    "eval_y_range": (1.0, 14.0),
    "eval_z": 0.05,
    "detection_threshold": 50.0,           # ppm-equivalent after 1e5 scaling
    "ood_split_key": "ventilation",        # absorption α mapped to ACH
    "ood_train_values": [6, 10],
    "ood_test_values": [3],
}

_CONFIGS = {
    "cfd": CFD_CONFIG,
    "acoustic": ACOUSTIC_CONFIG,
}


def get_domain_config(domain: str) -> dict:
    """Return the config dict for *domain* ('cfd' or 'acoustic')."""
    key = domain.lower()
    if key not in _CONFIGS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from {list(_CONFIGS)}")
    return _CONFIGS[key]
