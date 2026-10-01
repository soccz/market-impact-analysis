"""Use the previous immutable model requests and typed policy interpreter."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
PREVIOUS = ROOT.parent / "policy_compositional_reasoning"
sys.path.insert(0, str(PREVIOUS))
from infer import MODELS, sha, run as base_run
from infer_extended import request, run
from extended_logic import interpret, compile_program, variable, literal, FIELDS
from evaluate_capability import profile_exact, adapt
from evaluate import valid_direct
