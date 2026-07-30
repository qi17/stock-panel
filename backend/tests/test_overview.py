from datetime import date
import polars as pl
import pytest

from app.services.screener import ScreenerService
from app.services.market_overview_builder import build_market_overview, _index_quotes


class DummyStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir


class DummyRepo:
    def __init__(self, data_dir):
        self.store = DummyStore(data_dir)
        self._enriched_cache = None
        self._enriched_cache_date = None
        self._instruments_cache = pl.DataFrame()
        self.db_rows = []

    def get_enriched_latest_asset(self, asset_type="stock"):
        return self._enriched_cache, self._enriched_cache_date

    def get_instruments_asset(self, asset_type="stock"):
        return self._instruments_cache

    def get_enriched_history(self, target_date, lookback_days):
        return None

    def execute_all(self, query, params=None):
        return self.db_rows


def test_screener_load_enriched_filters_strictly_by_target_date(tmp_path):
    repo = DummyRepo(tmp_path)
    # Simulate a cache containing mixed dates for latest date 2026-07-27
    df_mixed = pl.DataFrame([
        {"symbol": "000001.SZ", "date": date(2026, 7, 27), "close": 10.0, "change_pct": 0.02, "volume": 100},
        {"symbol": "000002.SZ", "date": date(2026, 7, 26), "close": 15.0, "change_pct": 0.05, "volume": 200},
    ])
    repo._enriched_cache = df_mixed
    repo._enriched_cache_date = date(2026, 7, 27)

    svc = ScreenerService(repo)
    df_loaded = svc._load_enriched_for_date(date(2026, 7, 27))

    assert len(df_loaded) == 1
    assert df_loaded["symbol"][0] == "000001.SZ"
    assert df_loaded["date"][0] == date(2026, 7, 27)


def test_build_market_overview_uses_clean_date_data(tmp_path):
    repo = DummyRepo(tmp_path)
    # Today: 000001 is down -2%
    df_today = pl.DataFrame([
        {"symbol": "000001.SZ", "date": date(2026, 7, 27), "close": 10.0, "change_pct": -0.02, "amount": 1000.0, "volume": 100},
    ])
    repo._enriched_cache = df_today
    repo._enriched_cache_date = date(2026, 7, 27)

    overview = build_market_overview(repo, as_of=date(2026, 7, 27))
    assert overview["as_of"] == "2026-07-27"
    assert overview["breadth"]["total"] == 1
    assert overview["breadth"]["down"] == 1
    assert overview["breadth"]["up"] == 0


def test_index_quotes_db_fallback_handles_older_dates(tmp_path):
    repo = DummyRepo(tmp_path)
    # DB has latest index date 2026-07-24 (last_price 3814.2, prev_close 3876.7)
    repo.db_rows = [
        ("000001.SH", date(2026, 7, 24), 3814.2, 3876.7),
        ("399001.SZ", date(2026, 7, 24), 13774.6, 14123.0),
        ("399006.SZ", date(2026, 7, 24), 3480.8, 3575.5),
        ("000680.SH", date(2026, 7, 24), 1941.0, 1961.6),
    ]
    # Querying for target date 2026-07-27 should not return 2026-07-24's change pct as 07-27's change
    res = _index_quotes(repo, quote_service=None, as_of=date(2026, 7, 27))
    for item in res:
        assert item["change_pct"] == 0.0
