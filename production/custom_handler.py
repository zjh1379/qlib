"""Alpha158 子类：不同持有期版本。

为什么这么写？
    Alpha158 的 label 在 get_label_config() 里硬编码，不是 kwarg。
    直接改 site-packages\qlib\contrib\data\handler.py 是反模式
    （pip upgrade 会覆盖、跨项目污染）。
    继承一下覆盖 get_label_config 是干净的方式。

公式规律：N 日持有 → Ref($close, -(N+1)) / Ref($close, -1) - 1
"""
from qlib.contrib.data.handler import Alpha158, Alpha360


class Alpha158_3d(Alpha158):
    """3 天持有：T+1 收盘买，T+4 收盘卖。"""
    def get_label_config(self):
        return ["Ref($close, -4)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158_5d(Alpha158):
    """5 天持有（约一周）。"""
    def get_label_config(self):
        return ["Ref($close, -6)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158_10d(Alpha158):
    """10 天持有（约两周）。"""
    def get_label_config(self):
        return ["Ref($close, -11)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158_20d(Alpha158):
    """20 天持有（约一月）。"""
    def get_label_config(self):
        return ["Ref($close, -21)/Ref($close, -1) - 1"], ["LABEL0"]


# ============================================================================
# Open-to-open multi-horizon labels (β phase, T8).
#
# The 'open-to-open' label matches manual retail execution: the user places a
# buy on day T+1 morning and a sell on day T+1+N morning. This is more honest
# than the close-to-close used by stock Alpha158.
#
# Formula: Ref($open, -(N+1)) / Ref($open, -1) - 1
#         ^^^^^^^^^^^^^^^^^^ price N days after the buy
#                           ^^^^^^^^^^^^^ price on the buy morning
# ============================================================================


class Alpha158_OpenH(Alpha158):
    """Alpha158 features with open-to-open N-day label.

    Pass horizon_days via the kwargs dict in YAML:
        kwargs:
          horizon_days: 5
    """

    def __init__(self, horizon_days: int = 5, **kwargs):
        self.horizon_days = horizon_days
        super().__init__(**kwargs)

    def get_label_config(self):
        n = self.horizon_days
        return [f"Ref($open, -{n + 1}) / Ref($open, -1) - 1"], ["LABEL0"]


class Alpha360_OpenH(Alpha360):
    """Alpha360 features with open-to-open N-day label."""

    def __init__(self, horizon_days: int = 5, **kwargs):
        self.horizon_days = horizon_days
        super().__init__(**kwargs)

    def get_label_config(self):
        n = self.horizon_days
        return [f"Ref($open, -{n + 1}) / Ref($open, -1) - 1"], ["LABEL0"]


# ---------------------------------------------------------------------------
# Short-term factor handler
# ---------------------------------------------------------------------------
# Dual-import: when qlib loads this via module_path="custom_handler" with
# production/ on sys.path, only `factors.short_term` resolves.
# When imported as `production.custom_handler` from tests/backend, only
# `production.factors.short_term` resolves. The try/except handles both.
try:
    from production.factors.short_term import short_term_factor_config
except ModuleNotFoundError:
    from factors.short_term import short_term_factor_config  # type: ignore[no-redef]


class AlphaShortTerm(Alpha158_OpenH):
    """Alpha158 (open-to-open label) + non-redundant short-term factors.

    Adds OVNGAP / AMT_SURGE / LIMITUP*_CNT20 etc. on top of the 158 features,
    for the LGBM tabular path. Neural factor injection is deferred (P3+).
    """

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields, extra_names = short_term_factor_config()
        return list(fields) + extra_fields, list(names) + extra_names
