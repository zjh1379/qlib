from production.rolling_train import progress_total
from production.walk_forward import HorizonConfig


def _cfg(model_ids_enabled):
    # Minimal stand-in: progress_total only reads .model_specs and .horizons.
    class _C:
        model_specs = [{"id": m, "enabled": en} for m, en in model_ids_enabled]
        horizons = [
            HorizonConfig(name="1d", train_years=3, valid_years=1, stack_years=1, test_weeks=1),
            HorizonConfig(name="5d", train_years=5, valid_years=1, stack_years=1, test_weeks=1),
            HorizonConfig(name="20d", train_years=7, valid_years=1, stack_years=1, test_weeks=1),
        ]
    return _C()


def test_progress_total_counts_enabled_models_plus_fixed_stages():
    # 3 enabled models + universe(1) + ensemble(1) + done(1) = 6
    assert progress_total(_cfg([("lgbm", True), ("alstm", True), ("tra", True)])) == 6


def test_progress_total_ignores_disabled_models():
    # 1 enabled + 3 fixed = 4
    assert progress_total(_cfg([("lgbm", True), ("alstm", False), ("tra", False)])) == 4
