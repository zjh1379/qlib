"""Non-redundant short-term alpha factors (pure qlib expressions, no extra data).
All trailing (Ref +k = past) -> no lookahead. See spec 2026-06-03-shortterm-factors."""
from __future__ import annotations


def short_term_factor_config() -> tuple[list[str], list[str]]:
    """Return (expression_fields, names) appended onto Alpha158 for LGBM."""
    fields = [
        "$open/Ref($close,1)-1",                          # overnight gap
        "Mean($open/Ref($close,1)-1, 5)",                 # 5d mean gap
        "Std($open/Ref($close,1)-1, 5)",                  # 5d gap vol
        "($vwap*$volume)/Mean($vwap*$volume, 20)",        # amount surge (money flow proxy)
        "Sum(Gt($close/Ref($close,1)-1, 0.095), 20)",     # ~10% limit-up touches / 20d
        "Sum(Gt($close/Ref($close,1)-1, 0.19), 20)",      # ~20% (ChiNext/STAR) touches / 20d
    ]
    names = ["OVNGAP", "OVNGAP_MA5", "OVNGAP_STD5",
             "AMT_SURGE", "LIMITUP10_CNT20", "LIMITUP20_CNT20"]
    return fields, names
