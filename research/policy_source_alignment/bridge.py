"""Reuse frozen requests, typed execution and claim grounding without editing them."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
OLD = ROOT.parent / "policy_compositional_reasoning"
FACT = ROOT.parent / "policy_fact_grounding"
sys.path.insert(0, str(FACT))
sys.path.insert(0, str(OLD))
import infer
from infer import MODELS, OPTIONS, obj, arr, enum, sha
from infer_extended import request as old_request
from extended_logic import interpret, compile_program, FIELDS, variable, literal
from evaluate_capability import adapt, profile_exact

PREVIOUS = OLD
from grounding import audit as fact_audit, correct as correct_facts, money
