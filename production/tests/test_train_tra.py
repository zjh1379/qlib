from production.train_tra import _load_tra_config


def test_tra_config_loads_hyperparameters():
    cfg = _load_tra_config()
    assert cfg["model"]["kwargs"]["n_epochs"] == 100
    assert cfg["model"]["kwargs"]["tra_config"]["num_states"] == 10
