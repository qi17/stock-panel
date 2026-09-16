"""Generic HTTP provider for custom market data sources."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from app.config import settings
from app.data_providers.custom.config import CustomSourceConfig, DatasetConfig
from app.data_providers.custom.mapper import apply_transforms, datetime_payload, extract_rows, map_rows
from app.data_providers.normalizer import normalize_adj_factors, normalize_daily
from app.tickflow.rate_limits import chunked, sleep_between_batches

logger = logging.getLogger(__name__)

_REQUIRED = {
    "daily": {"symbol", "date", "open", "high", "low", "close", "volume", "amount"},
    "adj_factor": {"symbol", "trade_date", "ex_factor"},
    "realtime": {"symbol", "last_price", "prev_close", "open", "high", "low", "volume"},
    "minute": {"symbol", "datetime", "open", "high", "low", "close", "volume", "amount"},
    # financial 字段由数据源决定, 只要求能映射出 symbol
    "financial": {"symbol"},
}


class GenericHTTPProvider:
    """HTTP-backed custom source. It only handles fetching and schema mapping."""

    def __init__(self, config: CustomSourceConfig) -> None:
        self.config = config
        self.name = config.name
        self._client = httpx.Client(timeout=180.0, limits=httpx.Limits(max_keepalive_connections=0))

    def close(self) -> None:
        self._client.close()

    def validate(self) -> list[str]:
        errors: list[str] = []
        for dataset, cfg in self.config.datasets.items():
            if not cfg.url:
                errors.append(f"{dataset}: url is required")
            required = _REQUIRED.get(dataset)
            if required:
                mapped = set(cfg.field_map.values())
                missing = sorted(required - mapped)
                if missing:
                    errors.append(f"{dataset}: missing mapped fields: {', '.join(missing)}")
        return errors

    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
    ) -> pl.DataFrame:
        cfg = self._dataset("daily")
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, cfg.batch)
        for i, chunk in enumerate(chunks):
            sleep_between_batches(i, cfg.rpm)
            rows = self._request_rows(cfg, symbols=chunk, start_time=start_time, end_time=end_time)
            df = self._mapped_frame(cfg, rows)
            df = normalize_daily(df, source=self.name)
            if not df.is_empty():
                frames.append(df)
            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
    ) -> pl.DataFrame:
        cfg = self._dataset("adj_factor")
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, cfg.batch)

        # 批量读取本地 kline_daily parquet，构建 {symbol: {date_str: close}} 供 tdx-api-server 直接使用，
        # 跳过其内部逐只股票向 TDX 拉取 K 线的网络请求，大幅提速
        local_closes: dict[str, dict[str, float]] = {}
        try:
            kline_dir = settings.data_dir / "kline_daily"
            if kline_dir.exists():
                # 增量同步（有 start_time）：只取 start_time 前 7 天起的收盘价，减少请求体积
                # 全量同步（start_time=None）：取全部本地数据
                import datetime as _dt
                lf = pl.scan_parquet(
                    str(kline_dir / "**" / "*.parquet"),
                    extra_columns="ignore",
                ).select(["symbol", "date", "close"])
                if start_time is not None:
                    cutoff = (start_time - _dt.timedelta(days=7)).date()
                    lf = lf.filter(pl.col("date") >= cutoff)
                df_all = lf.collect()
                for row in df_all.iter_rows(named=True):
                    sym = row["symbol"]
                    d = str(row["date"])[:10]
                    if sym not in local_closes:
                        local_closes[sym] = {}
                    local_closes[sym][d] = float(row["close"])
                logger.info("get_adj_factors: 本地 kline_daily 已加载 %d 只股票收盘价 (cutoff=%s)",
                            len(local_closes), str(start_time)[:10] if start_time else "全量")
        except Exception as e:
            logger.warning("get_adj_factors: 本地 kline_daily 读取失败，回退至 TDX K 线查询: %s", e)

        for i, chunk in enumerate(chunks):
            sleep_between_batches(i, cfg.rpm)
            # 构建该批次的 kline_closes 子集
            chunk_closes = {sym: local_closes[sym] for sym in chunk if sym in local_closes}
            override_body = {"kline_closes": chunk_closes} if chunk_closes else None
            logger.info(
                "adj_factor 批次 %d/%d | 股票数=%d | 时间范围=[%s ~ %s] | 本地收盘价命中=%d/%d",
                i + 1, len(chunks), len(chunk),
                str(start_time)[:10] if start_time else "-",
                str(end_time)[:10] if end_time else "-",
                len(chunk_closes), len(chunk),
            )
            try:
                rows = self._request_rows(cfg, symbols=chunk, start_time=start_time, end_time=end_time,
                                          override_body=override_body)
                df = self._mapped_frame(cfg, rows)
                df = normalize_adj_factors(df, source=self.name)
                if not df.is_empty():
                    frames.append(df)
            except Exception as e:
                logger.warning("adj_factor 批次 %d/%d 获取失败: %s", i + 1, len(chunks), e)
            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def get_realtime(self, symbols: list[str] | None = None) -> list[dict]:
        cfg = self._dataset("realtime")
        rows = self._request_rows(cfg, symbols=symbols)
        df = self._mapped_frame(cfg, rows)
        if df.is_empty():
            return []
        return df.to_dicts()

    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
    ) -> pl.DataFrame:
        cfg = self._dataset("minute")
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, cfg.batch)
        for i, chunk in enumerate(chunks):
            sleep_between_batches(i, cfg.rpm)
            rows = self._request_rows(cfg, symbols=chunk, start_time=start_time, end_time=end_time)
            df = self._mapped_frame(cfg, rows)
            df = self._normalize_minute(df)
            if not df.is_empty():
                frames.append(df)
            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def get_financials(
        self,
        table: str,
        symbols: list[str],
        latest_only: bool = True,  # noqa: ARG002
    ) -> pl.DataFrame:
        """拉取财务数据。table ∈ {metrics, income, balance_sheet, cash_flow}。

        custom 源用一个 'financial' dataset 配置覆盖 4 张表; 请求时把 table 作为参数传给上游,
        上游根据 table 返回对应数据。字段由数据源决定, 这里只确保有 symbol 列。
        """
        cfg = self._dataset("financial")
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, cfg.batch)
        for i, chunk in enumerate(chunks):
            sleep_between_batches(i, cfg.rpm)
            # 把 table 注入到请求参数 (上游据此区分 4 张表)
            extra_params = {**cfg.params, "table": table}
            extra_body = {**cfg.body, "table": table}
            rows = self._request_rows(
                cfg, symbols=chunk,
                override_params=extra_params, override_body=extra_body,
            )
            df = self._mapped_frame(cfg, rows)
            if not df.is_empty():
                frames.append(df)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    @staticmethod
    def _normalize_minute(df: pl.DataFrame) -> pl.DataFrame:
        """把映射后的 df 规范成 minute canonical 列。"""
        if df.is_empty():
            return df
        if "datetime" in df.columns and df.schema["datetime"] != pl.Datetime("us"):
            if df.schema["datetime"] in (pl.Utf8, pl.Categorical):
                df = df.with_columns(
                    pl.col("datetime")
                    .str.replace(" ", "T", literal=True)
                    .cast(pl.Datetime("us"), strict=False)
                )
            else:
                df = df.with_columns(pl.col("datetime").cast(pl.Datetime("us"), strict=False))
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        keep = [c for c in ("symbol", "datetime", "open", "high", "low", "close", "volume", "amount") if c in df.columns]
        return df.select(keep) if keep else pl.DataFrame()

    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        cfg = self._dataset(dataset)
        symbols = symbols or ["600000.SH"]
        
        start_time = None
        end_time = None
        if dataset in ("daily", "minute", "adj_factor"):
            from datetime import timedelta
            now = datetime.now()
            if dataset == "minute":
                start_time = now - timedelta(days=1)
            else:
                start_time = now - timedelta(days=30)
            end_time = now

        rows = self._request_rows(cfg, symbols=symbols, start_time=start_time, end_time=end_time)
        df = self._mapped_frame(cfg, rows)
        return {
            "provider": self.name,
            "dataset": dataset,
            "rows": len(rows),
            "columns": df.columns,
            "preview": df.head(5).to_dicts() if not df.is_empty() else [],
        }

    def _dataset(self, name: str) -> DatasetConfig:
        cfg = self.config.datasets.get(name)
        if not cfg:
            raise ValueError(f"Custom data source '{self.name}' does not configure dataset '{name}'")
        return cfg

    def _mapped_frame(self, cfg: DatasetConfig, rows: list[dict]) -> pl.DataFrame:
        df = map_rows(rows, cfg.field_map)
        return apply_transforms(df, cfg.transforms)

    def _request_rows(
        self,
        cfg: DatasetConfig,
        *,
        symbols: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        override_params: dict[str, Any] | None = None,
        override_body: dict[str, Any] | None = None,
    ) -> list[dict]:
        headers, auth_params = self._auth_parts()
        params = dict(cfg.params)
        params.update(auth_params)
        if override_params:
            params.update(override_params)
        body = dict(cfg.body)
        if override_body:
            body.update(override_body)
        if symbols:
            body[cfg.symbols_param] = symbols
            params.setdefault(cfg.symbols_param, ",".join(symbols))
        start_value = datetime_payload(start_time)
        end_value = datetime_payload(end_time)
        if start_value:
            body[cfg.start_param] = start_value
            params.setdefault(cfg.start_param, start_value)
        if end_value:
            body[cfg.end_param] = end_value
            params.setdefault(cfg.end_param, end_value)

        method = cfg.method.upper()
        request_kwargs: dict[str, Any] = {"headers": headers, "timeout": cfg.timeout}
        if method == "GET":
            request_kwargs["params"] = params
        else:
            request_kwargs["params"] = auth_params
            request_kwargs["json"] = body
            
        logger.info(
            "Custom Provider [%s] Request: %s %s | Params: %s | Body: %s", 
            self.name, method, cfg.url, request_kwargs.get("params"), request_kwargs.get("json")
        )
        
        resp = self._client.request(method, cfg.url, **request_kwargs)
        resp.raise_for_status()
        return extract_rows(resp.json(), cfg.response_path)

    def _auth_parts(self) -> tuple[dict[str, str], dict[str, str]]:
        auth = self.config.auth
        if auth.type == "none":
            return {}, {}
        token = _token_from_env(auth.token_env) if auth.token_env else None
        if not token:
            logger.warning("custom data source %s auth token is not set", self.name)
            return {}, {}
        if auth.type == "bearer":
            return {auth.header: f"Bearer {token}"}, {}
        if auth.type == "header":
            return {auth.header: token}, {}
        if auth.type == "query":
            return {}, {auth.param: token}
        return {}, {}


def _token_from_env(name: str | None) -> str | None:
    if not name:
        return None
    token = os.getenv(name)
    if token:
        return token
    candidates = [settings.data_dir.parent / ".env", Path.cwd() / ".env", Path.cwd().parent / ".env"]
    env_path = next((path for path in candidates if path.exists()), None)
    if env_path is None:
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001
        return None
    return None
