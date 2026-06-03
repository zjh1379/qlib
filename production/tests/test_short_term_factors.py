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
