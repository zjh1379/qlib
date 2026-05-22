"""Recorder evaluation pipeline — computes 8-metric scorecard, regime
breakdown, acceptance pass/fail, paired t-test comparison.

Wraps the existing production/metrics.py + production/validate_acceptance.py
helpers and exposes them via REST + a CLI (production/eval_recorder.py).
"""
