"""
backend/expected_results.py

Single source of truth for the real, already-recorded accuracy/macro-F1
each exported checkpoint must reproduce (see CLAUDE.md / results/modality_ablation.csv).
Shared by scripts/export_checkpoints.py (verifies at export time) and
backend/registry.py (verifies again at serve time) so a checkpoint can
never silently drift from the number this app claims for it.
"""
EXPECTED = {
    "T": {"accuracy": 0.4206, "macro_f1": 0.2941},
    "A": {"accuracy": 0.1495, "macro_f1": 0.0687},
    "V": {"accuracy": 0.1121, "macro_f1": 0.0361},
    "TA": {"accuracy": 0.4112, "macro_f1": 0.2466},
    "TV": {"accuracy": 0.3364, "macro_f1": 0.2537},
    "AV": {"accuracy": 0.1963, "macro_f1": 0.0959},
    "TAV": {"accuracy": 0.3458, "macro_f1": 0.2083},
    "MISA": {"accuracy": 0.333, "macro_f1": 0.295},
}
TOLERANCE = 0.005
