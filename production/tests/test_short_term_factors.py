import sys
from pathlib import Path

# Strip the repo root from sys.path so the installed qlib (with compiled
# C extensions) is used instead of the source tree. Mirrors the pattern in
# test_multi_horizon_labels.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_saved_sys_path = list(sys.path)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _REPO_ROOT]
import qlib  # noqa: F401  # populate sys.modules with the installed qlib
sys.path[:] = _saved_sys_path

from production.factors.short_term import short_term_factor_config


def test_returns_fields_and_names_aligned():
    fields, names = short_term_factor_config()
    assert len(fields) == len(names) == 6
    assert names == ["OVNGAP", "OVNGAP_MA5", "OVNGAP_STD5",
                     "AMT_SURGE", "LIMITUP10_CNT20", "LIMITUP20_CNT20"]


def test_overnight_gap_expr():
    fields, names = short_term_factor_config()
    assert fields[names.index("OVNGAP")] == "$open/Ref($close,1)-1"


def test_no_forward_refs():
    # No lookahead: every Ref uses a POSITIVE shift (past). A negative Ref
    # like Ref($x,-2) would peek into the future.
    import re
    fields, _ = short_term_factor_config()
    for f in fields:
        for m in re.findall(r"Ref\([^,]+,\s*(-?\d+)\)", f):
            assert int(m) > 0, f"forward Ref in {f!r}"


def test_alpha_short_term_appends_factors_to_alpha158():
    from production.custom_handler import AlphaShortTerm
    # get_feature_config needs no data load; bypass __init__ via __new__.
    h = AlphaShortTerm.__new__(AlphaShortTerm)
    fields, names = h.get_feature_config()
    # superset of Alpha158 (158) + our 6
    assert "OVNGAP" in names and "LIMITUP10_CNT20" in names
    assert len(names) >= 158 + 6
