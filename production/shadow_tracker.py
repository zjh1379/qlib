"""Shadow paper trading tracker.

Per spec section 8: every new candidate model trains alongside production for 4 weeks
as `shadow_v2_ensemble`. After 4 weeks, if shadow IR > prod IR + 0.5, swap.

State is persisted in `state_path` (default production/shadow_state.json).
Tracker is intentionally simple — no DB; restart-safe via the JSON file.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_log = logging.getLogger("shadow_tracker")

DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "shadow_state.json"
PROMOTION_WEEKS = 4
PROMOTION_IR_THRESHOLD = 0.5


@dataclass
class ShadowState:
    weeks_observed: int = 0
    ir_history: list[float] = field(default_factory=list)
    baseline_ir_history: list[float] = field(default_factory=list)


class ShadowTracker:
    def __init__(self, state_path: Path = DEFAULT_STATE_PATH):
        self.state_path = state_path
        self._state: dict[str, ShadowState] = self._load()

    def _load(self) -> dict[str, ShadowState]:
        if not self.state_path.exists():
            return {}
        with self.state_path.open() as f:
            raw = json.load(f)
        return {k: ShadowState(**v) for k, v in raw.items()}

    def _save(self) -> None:
        serial = {
            k: {
                "weeks_observed": v.weeks_observed,
                "ir_history": v.ir_history,
                "baseline_ir_history": v.baseline_ir_history,
            }
            for k, v in self._state.items()
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(serial, indent=2))

    def record_run(self, recorder_id: str, run_date: date, is_shadow: bool, ir: float | None = None) -> None:
        if not is_shadow:
            return
        st = self._state.setdefault(recorder_id, ShadowState())
        st.weeks_observed += 1
        if ir is not None:
            st.ir_history.append(ir)
        self._save()

    def record_baseline(self, recorder_id: str, run_date: date, ir: float) -> None:
        # Append baseline IR to the most recently observed shadow candidate
        if not self._state:
            return
        last_id = max(self._state, key=lambda k: self._state[k].weeks_observed)
        self._state[last_id].baseline_ir_history.append(ir)
        self._save()

    def get_state(self, recorder_id: str) -> dict:
        st = self._state.get(recorder_id, ShadowState())
        return {"weeks_observed": st.weeks_observed, "ir_history": st.ir_history}

    def evaluate_promotion(self, recorder_id: str) -> dict:
        st = self._state.get(recorder_id)
        if st is None or st.weeks_observed < PROMOTION_WEEKS:
            return {"promote": False, "reason": "insufficient_weeks", "ir_delta": None}
        n = min(len(st.ir_history), len(st.baseline_ir_history))
        if n == 0:
            return {"promote": False, "reason": "no_ir_data", "ir_delta": None}
        delta = (sum(st.ir_history[:n]) - sum(st.baseline_ir_history[:n])) / n
        return {
            "promote": delta > PROMOTION_IR_THRESHOLD,
            "reason": "ok" if delta > PROMOTION_IR_THRESHOLD else "ir_delta_too_small",
            "ir_delta": delta,
        }
