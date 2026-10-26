from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.experiment_platform.timeseries import fit_bsts_model, run_duration_power_simulation
from src.experiment_platform.core import format_business_date


def test_app_uses_portable_date_directives():
    # %-d and %#d depend on the host C library and break on Windows/macOS.
    source = Path("app.py").read_text()
    assert "%-d" not in source
    assert "%#d" not in source


def test_business_dates_do_not_call_platform_strftime():
    with patch.object(pd.Timestamp, "strftime", side_effect=ValueError("Invalid format string")):
        assert format_business_date("2025-07-06") == "Jul 06, 2025"
        assert format_business_date("2025-07-06", False) == "Jul 2025"
        assert format_business_date(pd.NaT) == "Not available"


def test_timeseries_landing_and_workflows_survive_timestamp_strftime_failure():
    app = AppTest.from_file("app.py", default_timeout=45).run()
    with patch.object(pd.Timestamp, "strftime", side_effect=ValueError("Invalid format string")):
        app.button(key="home_choose_timeseries").click().run()
        assert not app.exception
        assert app.button(key="ts_demo_plan")
        app.button(key="ts_demo_plan").click().run()
        assert not app.exception
        app.button(key="ts_switch_workflow").click().run()
        app.button(key="ts_demo_analyze").click().run()
        assert not app.exception


def test_timeseries_plan_failure_is_actionable_not_an_app_exception():
    source = Path("app.py").read_text()
    source = source.replace('init_state()\nwith st.sidebar:', 'def run_duration_power_simulation(*args, **kwargs):\n    raise RuntimeError("Model fit unavailable")\n\ninit_state()\nwith st.sidebar:')
    app = AppTest.from_string(source, default_timeout=45).run()
    app.session_state.main_page = "Time Series"
    app.session_state.ts_intent = "plan"
    app.session_state.ts_config.data_source = "Synthetic Demo"
    app.run()
    assert not app.exception
    app.button(key="ts_plan_calculate").click().run()
    assert not app.exception
    assert any("Could not calculate" in item.value for item in app.error)
    assert app.session_state.ts_duration_plan is None


def test_switching_timeseries_to_empty_upload_clears_prepared_data():
    app = AppTest.from_file("app.py", default_timeout=45).run()
    app.session_state.main_page = "Time Series"
    app.session_state.ts_intent = "plan"
    app.session_state.ts_config.data_source = "Synthetic Demo"
    app.run()
    assert not app.session_state.ts_prepared_df.empty
    source = next(radio for radio in app.radio if radio.label == "Data Source")
    source.set_value("Upload File").run()
    assert not app.exception
    assert app.session_state.ts_prepared_df.empty


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_models_reject_missing_outcomes_before_fitting(invalid):
    data = pd.DataFrame({"_date": pd.date_range("2024-01-01", periods=60, freq="W"), "_outcome": np.arange(60, dtype=float)})
    data.loc[10, "_outcome"] = invalid
    with pytest.raises(ValueError, match="missing"):
        fit_bsts_model(data, "2024-09-01", "Weekly")
    with pytest.raises(ValueError, match="missing"):
        run_duration_power_simulation(data, "Weekly", [], "None", False, .95, 10, "Relative Lift (%)")
