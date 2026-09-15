from streamlit.testing.v1 import AppTest


def test_saved_protection_is_visible_and_stopped_engine_is_explicit():
    app = AppTest.from_string('''
from portfolio.virtual_account import LivePortfolio
from dashboard.profit_protection import render_profit_protection
port = LivePortfolio(10000)
port.buy("DEMO", 100., quantity=10.)
port.protect_profit("DEMO", 101., observed_at=1.)
render_profit_protection(port)
''').run(timeout=20)
    assert not app.exception
    row = app.dataframe[0].value.iloc[0]
    assert row["Protection"] == "Armed"
    assert row["Best observed price"] == 101.
    assert 100. < row["Exit trigger"] < 101.
    assert "not being monitored" in app.warning[0].value
