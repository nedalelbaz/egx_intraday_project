"""
EGX Intraday Trading Assistant (Enhanced Version)

This script implements a comprehensive intraday trading assistant for the
Egyptian Exchange (EGX).  It is designed to operate entirely on free,
publicly‑available data sources and does not execute live trades.  Instead
it provides signals, risk management, journalling and analysis features
for a paper‑trading workflow.  The system includes the following key
capabilities:

1. **Data aggregation and normalisation** – The assistant pulls
   intraday price data from multiple free providers (Yahoo Finance,
   Twelve Data and optional community feeds) with automatic failover.
   Data are normalised into a common schema (`Timestamp`, `Open`, `High`,
   `Low`, `Close`, `Volume`) and are timestamped in the Cairo timezone.

2. **Opening Range Breakout strategy** – A classic ORB strategy is
   implemented with support for long and short trades.  Breakouts are
   confirmed using momentum indicators (EMA crossover and RSI), volume
   surges and multi‑timeframe verification.  Trailing stops, dynamic
   take‑profit levels and multiple entries per trade are supported.

3. **Risk management** – Position sizes are calculated based on a
   user‑configurable risk percentage of account equity.  Daily loss
   limits, capital protection buffers and profit lock rules are
   enforced.  Each day the system can withdraw a portion of profits,
   either a fixed amount or a percentage, and reinvest the remainder.
   Transaction costs (commissions and stamp taxes) are factored into
   PnL calculations.

4. **Journalling and analytics** – All signals, trades and misses
   are logged to a SQLite database.  For each closed trade the
   assistant records the maximum favourable excursion (MFE) and maximum
   adverse excursion (MAE).  Daily, weekly and monthly KPI reports are
   generated automatically and delivered via Telegram.  Historical
   reports are archived in CSV and PNG formats, with old reports
   compressed into ZIP files after 90 days.

5. **Machine learning scoring** – A logistic regression model is
   trained on past trades to estimate the probability of a new signal
   succeeding.  This probability is blended with the raw momentum
   confirmations to produce an overall confidence score for each
   signal.

6. **Interactive alerts** – Signals are pushed to Telegram with
   inline buttons for Buy, Skip or Adjust.  A background poller
   monitors user responses and logs them.  Optional chart images are
   attached to alerts.

7. **Reporting and backtesting hooks** – The script includes
   functions to generate simple backtests using historical intraday
   data and to produce weekly/monthly reports summarising performance.

This file is designed to be self‑contained; to customise behaviour edit
the accompanying `config.yaml`.  Running the script will start a loop
that fetches data during market hours, analyses signals, sends alerts
and handles journalling and reporting.

Author: OpenAI
Date: September 2025
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import math
import os
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import yaml
import concurrent.futures  # Added for concurrent data fetching

try:
    # Optional ML imports.  The logistic regression scorer will be
    # disabled if scikit‑learn is not available.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception:
    LogisticRegression = None  # type: ignore
    StandardScaler = None  # type: ignore
    Pipeline = None  # type: ignore

try:
    # Optional charting import.  If unavailable, chart images will be
    # skipped.  Matplotlib is used rather than seaborn to comply with
    # environment guidelines.
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore

try:
    # yfinance is used to fetch data from Yahoo Finance.
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

try:
    # mplfinance is used for candlestick chart generation.
    import mplfinance as mpf
except Exception:
    mpf = None  # type: ignore

CAIRO_TZ = pytz.timezone("Africa/Cairo")


###############################################################################
# Configuration
###############################################################################

DEFAULT_CONFIG = {
    "symbols": [
        "COMI.CA",  # Commercial International Bank
        "EGTS.CA",  # Egyptian Resorts
        "EFIH.CA",  # e-finance
        "ORHD.CA",  # Orascom Development Holding
        "EKHO.CA",  # EFG Hermes (EK Holding)
    ],
    "short": {
        "allow_short": False,
        # Specify a subset of symbols allowed to be shorted.  If empty
        # and allow_short is True then any symbol can be shorted.
        "short_list": [],
    },
    "sectors": {
        "COMI.CA": "Banks",
        "EGTS.CA": "Real Estate",
        "EFIH.CA": "Tech",
        "ORHD.CA": "Real Estate",
        "EKHO.CA": "Financials",
    },
    "market": {
        "open_time": "10:00",        # Cairo time
        "close_time": "14:30",
        "open_range_minutes": 15,     # ORB window length
        "late_wake_grace_min": 90,    # Start late but still compute ORB
    },
    "data": {
        "providers_order": ["yahoo", "twelvedata", "egxlytics", "investing"],
        "interval": "1m",            # Interval for intraday data
        "lookback_minutes": 240,      # Number of minutes to fetch
        "twelvedata_api_key": "",     # Twelve Data API key (optional)
        # News RSS feeds (optional).  Add URLs for EGX, Mubasher or
        # Investing.com feeds if available.  Sentiment analysis will be
        # performed on headlines.
        "rss_feeds": [],
        "news_max_items": 20,
    },
    "strategy": {
        "momentum": {
            "ema_fast": 9,
            "ema_slow": 21,
            "rsi_len": 14,
            "rsi_threshold": 55,
        },
        "volume": {
            "vol_ma_len": 20,
            "surge_factor": 1.8,
        },
        "multi_timeframes": ["5min", "15min", "60min"],
        # Risk buffers are multiples of ATR.  Primary buffer sets the
        # initial stop‑loss distance; reserve buffer triggers a second
        # chance entry; trailing buffer controls the trailing stop
        # distance.  For short trades, buffers are applied symmetrically.
        "primary_buffer_atr_mult": 0.15,
        "reserve_buffer_atr_mult": 0.10,
        "trailing_atr_mult": 0.8,
        # Reward multiple for take profit relative to risk.  If
        # risk (distance between entry and stop) is R, then take
        # profit is at entry ± R*take_profit_rr.
        "take_profit_rr": 2.0,
        "allow_multi_entry": True,
        "max_adds": 2,
        "min_confidence_for_alert": 0.6,
        "late_wake_mode": True,
    },
    "risk": {
        "daily_start_capital": 100000.0,
        "capital_protection_buffer": 0.85,  # Stop trading if equity < 85% of start
        "daily_loss_limit": 0.03,            # Stop if daily loss > 3%
        "profit_lock_trigger": 0.02,         # If PnL > 2% then lock gains
        "profit_lock_giveback": 0.5,         # Giveback fraction for profit lock
        # Withdrawal settings.  At end of profitable day withdraw either
        # a fixed amount or a percentage of profits.  Only used if
        # profits > 0.
        "withdrawal_mode": "fixed",         # "fixed" or "percent" or "hybrid"
        "daily_withdrawal": 1500.0,          # Fixed amount (EGP)
        "withdrawal_percent": 0.3,          # Percentage of profit to withdraw
        # Transaction cost model: commission and stamp tax on each side.
        # Commission rate and stamp tax rate are fractions (e.g. 0.0015
        # equals 0.15%).  These costs are deducted from PnL for each
        # trade (entry + exit).  Adjust according to broker fees and
        # Egyptian stamp tax rules.  Leave as zero to ignore costs.
        "transaction_cost": {
            "commission_rate": 0.0010,
            "stamp_tax_rate": 0.0015,
        },
        # Position sizing.  base_risk_per_trade is expressed as a
        # percentage of account equity.  volatility_scale_cap limits
        # position sizes in quiet markets.  confidence_scale increases
        # size when the ML confidence is high.
        "position_sizing": {
            "base_risk_per_trade": 0.5,    # 0.5% of equity per trade
            "volatility_scale_cap": 1.75,
            "confidence_scale": True,
        },
        "extreme_reserve_fraction": 0.1,     # Fraction of capital kept aside
        "max_trades_per_day": 10,            # Maximum trades in a day (long+short)
    },
    "alerts": {
        "telegram_bot_token": "",    # Provide your Telegram bot token
        "telegram_chat_id": "",      # Provide your chat ID
        "send_charts": True,
        "filter_high_confidence": True,
    },
    "journal": {
        "db_path": "egx_intraday.sqlite",
        "report_out_dir": "reports",
    },
    "watchlist": {
        # When generating a daily watchlist, rank stocks by a weighted
        # combination of gap %, volume surge and momentum slope.
        "top_n": 10,
        "rank_features": {
            "gap_weight": 0.4,
            "vol_surge_weight": 0.3,
            "momentum_weight": 0.3,
        },
    },
}


###############################################################################
# Utility Functions
###############################################################################


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file or create a default one if missing."""
    if not os.path.exists(path):
        # Write default config for the user to edit on first run.
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, sort_keys=False, allow_unicode=True)
        print(f"Created default {path}. Please review & rerun.")
        sys.exit(0)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def now_cairo() -> dt.datetime:
    """Return current time in Cairo timezone."""
    return dt.datetime.now(CAIRO_TZ)


def to_cairo(ts: dt.datetime) -> dt.datetime:
    """Convert timestamp to Cairo timezone if not already localised."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=pytz.UTC).astimezone(CAIRO_TZ)
    return ts.astimezone(CAIRO_TZ)


def parse_time_local(tstr: str) -> dt.time:
    """Parse a HH:MM string into a dt.time in Cairo timezone."""
    hh, mm = map(int, tstr.split(":"))
    return dt.time(hh, mm)


def between_market_hours(ts: dt.datetime, cfg: dict) -> bool:
    """Return True if the timestamp is within market hours."""
    open_t = parse_time_local(cfg["market"]["open_time"])
    close_t = parse_time_local(cfg["market"]["close_time"])
    local_ts = to_cairo(ts)
    return open_t <= local_ts.time() <= close_t


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and ensure required columns exist.

    The returned DataFrame will have columns: Timestamp, Open,
    High, Low, Close, Volume.  Any missing columns are filled with NaN.
    Timestamps are converted to timezone‑aware datetimes in Cairo.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    # Normalise column names (case insensitive)
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        low = col.lower()
        if low in ("date", "datetime", "timestamp"):
            rename_map[col] = "Timestamp"
        elif low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low in ("close", "adj close", "price"):
            rename_map[col] = "Close"
        elif low in ("volume", "vol"):
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)
    # Ensure required columns
    for col in ["Timestamp", "Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    # Convert timestamp column to datetime with timezone
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df["Timestamp"] = df["Timestamp"].dt.tz_convert(CAIRO_TZ)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def resample_agg(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV DataFrame to a given rule (e.g. '5min')."""
    if df.empty:
        return df
    out = df.set_index("Timestamp").resample(rule, origin="start_day", offset="0min").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna().reset_index()
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def atr_like(df: pd.DataFrame, length: int = 14) -> float:
    """Compute a simple ATR‑like measure on intraday data.

    Because traditional ATR uses previous day data, we approximate by
    computing the rolling average of the true range across the last
    `length` periods.  True range is the maximum of: high‑low,
    absolute(high‑previous close) and absolute(low‑previous close).
    """
    if df.empty or len(df) < max(length, 2):
        return float("nan")
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
    atr = tr.rolling(length).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else float("nan")


###############################################################################
# Data Providers
###############################################################################


class BaseProvider:
    """Abstract base class for data providers."""
    name: str = "base"

    def fetch_intraday(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        raise NotImplementedError


class YahooProvider(BaseProvider):
    """Fetch intraday data from Yahoo Finance using yfinance."""
    name = "yahoo"

    def fetch_intraday(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()
        # Determine appropriate period.  Yahoo restricts periods per interval.
        period_map = {
            "1m": "1d",
            "2m": "1d",
            "5m": "5d",
            "15m": "5d",
            "60m": "1mo",
        }
        period = period_map.get(interval, "1d")
        try:
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=False,
                prepost=False,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.reset_index().rename(columns={"Datetime": "Timestamp"})
            df = ensure_cols(df)
            # Clip to lookback window
            cutoff = now_cairo() - dt.timedelta(minutes=lookback_minutes + 5)
            df = df[df["Timestamp"] >= cutoff]
            return df
        except Exception as e:
            logging.getLogger("EGXIntraday").warning(f"Yahoo fetch failed for {symbol}: {e}")
            return pd.DataFrame()


class TwelveDataProvider(BaseProvider):
    """Fetch intraday data from Twelve Data API."""
    name = "twelvedata"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_intraday(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame()
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": min(5000, lookback_minutes + 30),
            "timezone": "Africa/Cairo",
            "format": "JSON",
            "apikey": self.api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return pd.DataFrame()
            data = r.json()
            if not data or "values" not in data:
                return pd.DataFrame()
            df = pd.DataFrame(data["values"])
            df = df.rename(columns={
                "datetime": "Timestamp",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            })
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.tz_localize(CAIRO_TZ, nonexistent="shift_forward", ambiguous="NaT")
            df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
            cutoff = now_cairo() - dt.timedelta(minutes=lookback_minutes + 5)
            df = df[df["Timestamp"] >= cutoff]
            return df
        except Exception as e:
            logging.getLogger("EGXIntraday").warning(f"TwelveData fetch failed for {symbol}: {e}")
            return pd.DataFrame()


class EGXlyticsProvider(BaseProvider):
    """Placeholder for EGXlytics (community or personal feed)."""
    name = "egxlytics"

    def fetch_intraday(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        # This provider should be implemented by the user if they have a
        # local or community feed.  Return an empty DataFrame by default.
        return pd.DataFrame()


class InvestingProvider(BaseProvider):
    """Placeholder for Investing.com or similar scraped data."""
    name = "investing"

    def fetch_intraday(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        # Not implemented: left for user customisation.  Some community
        # projects provide APIs to Investing.com; due to TOS restrictions
        # this is left blank.  Return empty DataFrame.
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# DirectFN / Mubasher provider
#
# This provider wraps the DirectFN/Mubasher "today traded" or intraday bars
# API.  DirectFN offers delayed or real‑time market data via a web‑service
# endpoint (JSON or XML) for EGX.  The same endpoint can be upgraded from
# delayed to real‑time by license without code changes.  This class accepts
# a base URL and API key plus optional endpoint paths and symbol format.  It
# emits data in the same schema used throughout this assistant: Timestamp,
# Open, High, Low, Close, Volume and Provider.  Resampling is performed if
# the feed returns trade snapshots rather than aggregated bars.

class DirectFNProvider(BaseProvider):
    """Fetch intraday data from DirectFN / Mubasher.

    The provider supports both delayed and real‑time feeds.  It can build
    minute bars from the "today traded" snapshot endpoint when an explicit
    intraday bars endpoint is not available.  Timestamps are localised to
    Africa/Cairo by default.
    """
    name = "directfn"

    def __init__(self,
                 base_url: str,
                 api_key: str,
                 today_traded_path: str = "/api/v1/todaytraded",
                 intraday_path: str = "",
                 symbol_format: str = "plain",
                 tz: str = "Africa/Cairo",
                 timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.today_traded_path = today_traded_path
        self.intraday_path = intraday_path
        self.symbol_format = symbol_format
        self.tz = pytz.timezone(tz)
        self.timeout = timeout

    def _fmt_symbol(self, sym: str) -> str:
        return f"{sym}.EGX" if self.symbol_format == "egx_suffix" else sym

    def _to_df(self, rows: List[dict]) -> pd.DataFrame:
        """Convert vendor payload to DataFrame with standard columns."""
        records = []
        for r in rows:
            # Timestamps may be ISO strings or epoch seconds
            ts = r.get("datetime") or r.get("ts") or r.get("time")
            if ts is None:
                continue
            try:
                if isinstance(ts, (int, float)):
                    dt_local = dt.datetime.fromtimestamp(int(ts), tz=self.tz)
                else:
                    dt_local = dt.datetime.fromisoformat(str(ts))
                    if dt_local.tzinfo is None:
                        dt_local = self.tz.localize(dt_local)
                    else:
                        dt_local = dt_local.astimezone(self.tz)
            except Exception:
                continue
            # Map vendor fields to OHLCV; fallback to last price if open/high/low missing
            last = float(r.get("last_trade_price", r.get("close", r.get("last", 0))))
            o = float(r.get("open", last))
            h = float(r.get("high", last))
            l = float(r.get("low", last))
            v = float(r.get("volume", r.get("qty", 0)))
            records.append({
                "Timestamp": dt_local,
                "Open": o,
                "High": h,
                "Low": l,
                "Close": last,
                "Volume": v,
                "Provider": self.name,
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("Timestamp").reset_index(drop=True)
        return df

    def fetch_intraday(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        # Require an API key; return empty if not provided
        if not self.api_key or not self.base_url:
            return pd.DataFrame()
        sym = self._fmt_symbol(symbol)
        # If an intraday bars endpoint is provided, use it directly
        if self.intraday_path:
            url = f"{self.base_url}{self.intraday_path}"
            params = {
                "symbol": sym,
                "interval": interval,
                "apikey": self.api_key,
                "lookback": lookback_minutes,
            }
            try:
                r = requests.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                rows = data.get("rows", data) if isinstance(data, dict) else data
                return self._to_df(rows)
            except Exception as e:
                logging.getLogger("EGXIntraday").warning(f"DirectFN intraday fetch failed for {symbol}: {e}")
                return pd.DataFrame()
        # Otherwise build minute bars from today traded snapshot
        url = f"{self.base_url}{self.today_traded_path}"
        params = {"symbol": sym, "apikey": self.api_key}
        try:
            r = requests.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            rows = data.get("rows", data) if isinstance(data, dict) else data
            df = self._to_df(rows)
            # Resample to interval if the feed isn't aggregated
            if df.empty:
                return df
            if interval.endswith("m"):
                # Determine minutes to resample (e.g. "1m" -> 1)
                try:
                    mins = int(interval[:-1])
                except Exception:
                    mins = 1
                g = df.set_index("Timestamp").resample(f"{mins}T", label="right", closed="right")
                df = pd.DataFrame({
                    "Open": g["Open"].first(),
                    "High": g["High"].max(),
                    "Low": g["Low"].min(),
                    "Close": g["Close"].last(),
                    "Volume": g["Volume"].sum(),
                }).dropna(how="all").reset_index()
                df["Provider"] = self.name
            return df
        except Exception as e:
            logging.getLogger("EGXIntraday").warning(f"DirectFN todaytraded fetch failed for {symbol}: {e}")
            return pd.DataFrame()


class DataRouter:
    """Route data fetch to multiple providers with failover."""

    def __init__(self, cfg: dict):
        self.providers: List[BaseProvider] = []
        order = cfg["data"]["providers_order"]
        for p in order:
            if p == "yahoo":
                self.providers.append(YahooProvider())
            elif p == "twelvedata":
                self.providers.append(TwelveDataProvider(cfg["data"].get("twelvedata_api_key", "")))
            elif p == "egxlytics":
                self.providers.append(EGXlyticsProvider())
            elif p == "investing":
                self.providers.append(InvestingProvider())
            elif p == "directfn":
                # Only construct DirectFNProvider if config provides API key and base URL
                df_cfg = cfg["data"].get("directfn", {})
                api_key = df_cfg.get("api_key")
                base_url = df_cfg.get("base_url")
                if api_key and base_url:
                    self.providers.append(
                        DirectFNProvider(
                            base_url=base_url,
                            api_key=api_key,
                            today_traded_path=df_cfg.get("today_traded_path", "/api/v1/todaytraded"),
                            intraday_path=df_cfg.get("intraday_path", ""),
                            symbol_format=df_cfg.get("symbol_format", "plain"),
                            tz=cfg.get("tz", "Africa/Cairo"),
                            timeout=int(df_cfg.get("timeout", 15)),
                        )
                    )

    def fetch(self, symbol: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
        """Attempt to fetch data from providers in order until success."""
        for prov in self.providers:
            df = prov.fetch_intraday(symbol, interval, lookback_minutes)
            if df is not None and not df.empty:
                df["Provider"] = prov.name
                return df
        return pd.DataFrame()


###############################################################################
# Telegram Messaging
###############################################################################


class Telegram:
    """Wrapper around Telegram Bot API for sending messages and images."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{bot_token}"
        self.logger = logging.getLogger("EGXIntraday.Telegram")

    def send_message(self, text: str, buttons: Optional[List[List[Dict]]] = None, disable_web_page_preview: bool = True) -> None:
        if not self.bot_token or not self.chat_id:
            self.logger.info(f"Telegram disabled: {text}")
            return
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_web_page_preview,
        }
        if buttons:
            payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})
        try:
            requests.post(f"{self.base}/sendMessage", data=payload, timeout=10)
        except Exception as e:
            self.logger.warning(f"Telegram send error: {e}")

    def send_photo(self, caption: str, image_bytes: bytes) -> None:
        if not self.bot_token or not self.chat_id:
            self.logger.info("Telegram disabled: chart image not sent")
            return
        files = {"photo": ("chart.png", image_bytes)}
        data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"}
        try:
            requests.post(f"{self.base}/sendPhoto", data=data, files=files, timeout=15)
        except Exception as e:
            self.logger.warning(f"Telegram photo error: {e}")

    def send_plot(self, fig, caption: str = "") -> None:
        """Send a matplotlib figure as a Telegram photo."""
        if plt is None:
            self.logger.info("Matplotlib not available: cannot send plot")
            return
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        self.send_photo(caption, buf.read())


###############################################################################
# Journal and Database
###############################################################################


class Journal:
    """Journal for recording alerts, trades, misses and KPIs."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        self.logger = logging.getLogger("EGXIntraday.Journal")

    def _init_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        # Enable write‑ahead logging for concurrency and to avoid database
        # lock errors.  If WAL mode is already enabled the pragma has
        # no effect.  We execute this pragma up front before table
        # creation to ensure it persists.
        try:
            con.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            # On some environments WAL may be unsupported; ignore
            pass
        cur = con.cursor()
        # Alerts table logs every alert sent to Telegram
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                symbol TEXT,
                side TEXT,
                reason TEXT,
                confidence REAL,
                price REAL,
                orb_high REAL,
                orb_low REAL,
                provider TEXT,
                chart_sent INTEGER DEFAULT 0
            )
            """
        )
        # Trades table logs entry and exit information
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_open TEXT,
                ts_close TEXT,
                symbol TEXT,
                side TEXT,
                entry REAL,
                exit REAL,
                qty REAL,
                pnl REAL,
                fees REAL,
                mfe REAL,
                mae REAL,
                adds INTEGER,
                notes TEXT
            )
            """
        )
        # Misses table logs when data is missing or user actions (e.g. skip)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS misses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                symbol TEXT,
                reason TEXT
            )
            """
        )
        # Missed opportunities table logs signals that were generated but not executed
        # due to risk brakes, trade limits or other constraints.  Stores the
        # hypothetical entry information along with ORB bounds and will be
        # updated with potential PnL at end of day.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS missed_opps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                symbol TEXT,
                side TEXT,
                price REAL,
                qty REAL,
                orb_high REAL,
                orb_low REAL,
                reason TEXT,
                potential_pnl REAL,
                pot_ts_close TEXT
            )
            """
        )
        con.commit()
        con.close()

    def log_alert(self, **kwargs) -> None:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO alerts (ts, symbol, side, reason, confidence, price, orb_high, orb_low, provider, chart_sent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kwargs.get("ts"),
                kwargs.get("symbol"),
                kwargs.get("side"),
                kwargs.get("reason"),
                kwargs.get("confidence"),
                kwargs.get("price"),
                kwargs.get("orb_high"),
                kwargs.get("orb_low"),
                kwargs.get("provider"),
                1 if kwargs.get("chart_sent") else 0,
            ),
        )
        con.commit()
        con.close()

    def log_trade(self, **kw) -> None:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO trades (ts_open, ts_close, symbol, side, entry, exit, qty, pnl, fees, mfe, mae, adds, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kw.get("ts_open"),
                kw.get("ts_close"),
                kw.get("symbol"),
                kw.get("side"),
                kw.get("entry"),
                kw.get("exit"),
                kw.get("qty"),
                kw.get("pnl"),
                kw.get("fees"),
                kw.get("mfe"),
                kw.get("mae"),
                kw.get("adds"),
                kw.get("notes"),
            ),
        )
        con.commit()
        con.close()

    def log_miss(self, **kw) -> None:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO misses (ts, symbol, reason)
            VALUES (?, ?, ?)
            """,
            (
                kw.get("ts"),
                kw.get("symbol"),
                kw.get("reason"),
            ),
        )
        con.commit()
        con.close()

    def log_missed_opportunity(self, ts: str, symbol: str, side: str, price: float, qty: float, orb_high: float, orb_low: float, reason: str) -> None:
        """Record a missed opportunity when a signal is not executed."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO missed_opps (ts, symbol, side, price, qty, orb_high, orb_low, reason, potential_pnl, pot_ts_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, symbol, side, price, qty, orb_high, orb_low, reason, None, None),
        )
        con.commit()
        con.close()

    def fetch_missed_opps(self, day: dt.date) -> pd.DataFrame:
        """Return all missed opportunities for a given day."""
        con = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM missed_opps", con, parse_dates=["ts", "pot_ts_close"])
        con.close()
        if df.empty:
            return df
        df["day"] = df["ts"].dt.tz_localize("UTC").dt.tz_convert(CAIRO_TZ).dt.date
        return df[df["day"] == day]

    def update_missed_pnl(self, row_id: int, potential_pnl: float, ts_close: dt.datetime) -> None:
        """Update a missed opportunity record with its computed potential PnL."""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            "UPDATE missed_opps SET potential_pnl = ?, pot_ts_close = ? WHERE id = ?",
            (potential_pnl, ts_close.isoformat() if isinstance(ts_close, dt.datetime) else ts_close, row_id),
        )
        con.commit()
        con.close()

    def daily_report(self, day: dt.date) -> Dict[str, Optional[float]]:
        """Compute daily KPIs from trade logs."""
        con = sqlite3.connect(self.db_path)
        df_tr = pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["ts_open", "ts_close"])
        con.close()
        if df_tr.empty:
            return {"hit_rate": None, "net_pnl": 0.0, "num_trades": 0, "best_symbol": None, "worst_symbol": None}
        df_tr["day"] = df_tr["ts_open"].dt.tz_localize("UTC").dt.tz_convert(CAIRO_TZ).dt.date
        d = df_tr[df_tr["day"] == day]
        if d.empty:
            return {"hit_rate": None, "net_pnl": 0.0, "num_trades": 0, "best_symbol": None, "worst_symbol": None}
        wins = (d["pnl"] > 0).sum()
        hit_rate = wins / len(d)
        agg = d.groupby("symbol")["pnl"].sum().sort_values()
        best = agg.iloc[-1:].index[0] if len(agg) > 0 else None
        worst = agg.iloc[:1].index[0] if len(agg) > 0 else None
        return {
            "hit_rate": hit_rate,
            "net_pnl": float(d["pnl"].sum()),
            "num_trades": int(len(d)),
            "best_symbol": best,
            "worst_symbol": worst,
        }


###############################################################################
# Machine Learning Scorer
###############################################################################


class SignalScorer:
    """Logistic regression model to estimate trade success probability."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model: Optional[Pipeline] = None
        self.logger = logging.getLogger("EGXIntraday.ML")
        # Train model on initialisation if scikit‑learn is available
        if LogisticRegression is not None:
            self.train()
        else:
            self.logger.info("scikit‑learn not available; ML scoring disabled")

    def load_trades(self) -> pd.DataFrame:
        con = sqlite3.connect(self.db_path)
        # Load executed trades
        trades_df = pd.read_sql_query("SELECT * FROM trades", con)
        # Load missed opportunities with computed potential_pnl
        miss_df = pd.read_sql_query("SELECT * FROM missed_opps WHERE potential_pnl IS NOT NULL", con)
        con.close()
        # Prepare missed opportunities to align with trades_df and retain all useful columns
        if not miss_df.empty:
            miss_df = miss_df.copy()
            # Use potential_pnl as pnl for training label
            miss_df["pnl"] = miss_df["potential_pnl"].astype(float)
            # Define entry and exit columns for feature extraction
            miss_df["entry"] = miss_df["price"]
            miss_df["exit"] = miss_df["price"]
            miss_df["qty"] = miss_df["qty"].fillna(0.0)
            miss_df["adds"] = 0
        # Create unified column set
        all_cols = set(trades_df.columns) | set(miss_df.columns)
        # Ensure both DataFrames have all columns
        for col in all_cols:
            if not trades_df.empty and col not in trades_df.columns:
                trades_df[col] = None
            if not miss_df.empty and col not in miss_df.columns:
                miss_df[col] = None
        # Concatenate if both non‑empty
        if not trades_df.empty and not miss_df.empty:
            return pd.concat([trades_df[all_cols], miss_df[all_cols]], ignore_index=True, sort=False)
        elif not trades_df.empty:
            return trades_df[all_cols]
        elif not miss_df.empty:
            return miss_df[all_cols]
        else:
            return pd.DataFrame()

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML model from trades and missed opportunities.

        For executed trades, the risk feature is computed as the difference
        between entry and exit price (entry minus exit).  For missed
        opportunities (identified by presence of the potential_pnl column),
        risk is computed using the ORB bounds: for long signals (side BUY/LONG)
        it is price minus orb_low; for short signals (side SELL/SHORT) it is
        orb_high minus price.  If ORB information is missing, risk defaults
        to zero.  Additional features include number of adds, entry price and
        quantity.
        """
        # Ensure DataFrame is not empty
        if df.empty:
            return pd.DataFrame()
        # Initialise features DataFrame with same index
        feats = pd.DataFrame(index=df.index)
        risk_vals: List[float] = []
        for idx, row in df.iterrows():
            # Determine if this row has potential_pnl (missed opportunity)
            is_missed = pd.notna(row.get("potential_pnl"))
            if is_missed:
                price = row.get("entry", row.get("price", 0.0))
                side = row.get("side", "BUY")
                orb_low = row.get("orb_low")
                orb_high = row.get("orb_high")
                # Compute risk using ORB bounds when available
                if pd.notna(orb_low) and pd.notna(orb_high):
                    if str(side).upper() in ("BUY", "LONG"):
                        risk = float(price) - float(orb_low)
                    else:
                        risk = float(orb_high) - float(price)
                else:
                    risk = 0.0
            else:
                # Executed trade: risk as entry minus exit
                entry_price = row.get("entry", 0.0)
                exit_price = row.get("exit", entry_price)
                risk = float(entry_price) - float(exit_price)
            risk_vals.append(risk)
        feats["risk"] = pd.Series(risk_vals, index=df.index)
        feats["adds"] = df.get("adds", 0).fillna(0)
        # Use entry price for both executed and missed (for missed, entry is price)
        feats["entry"] = df.get("entry", df.get("price", 0)).fillna(0)
        feats["qty"] = df.get("qty", 0).fillna(0)
        return feats.fillna(0)

    def train(self) -> None:
        if LogisticRegression is None:
            return
        df = self.load_trades()
        if df.empty:
            return
        # Outcome: 1 if pnl > 0 else 0
        y = (df["pnl"] > 0).astype(int)
        X = self.featurize(df)
        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ])
            pipe.fit(X, y)
            self.model = pipe
            self.logger.info("ML scorer trained on %d trades", len(df))
        except Exception as e:
            self.logger.warning(f"ML training failed: {e}")
            self.model = None

    def score(self, features: dict) -> float:
        """Return success probability estimate for a signal based on its features."""
        if self.model is None:
            return 0.5
        X = pd.DataFrame([features])
        try:
            proba = self.model.predict_proba(X)[0, 1]
            return float(proba)
        except Exception as e:
            self.logger.warning(f"ML scoring failed: {e}")
            return 0.5


###############################################################################
# Trading Entities
###############################################################################


@dataclass
class Signal:
    symbol: str
    ts: dt.datetime
    side: str              # "BUY" for long, "SELL" for short (i.e. short entry)
    reason: str
    confidence: float
    price: float
    orb_high: float
    orb_low: float
    provider: str
    chart: Optional[bytes] = None


@dataclass
class Position:
    symbol: str
    side: str              # "LONG" or "SHORT"
    entry: float
    qty: float
    ts_open: dt.datetime
    orb_high: float
    orb_low: float
    buffers: Dict[str, float]
    adds_done: int = 0
    trail_stop: Optional[float] = None
    mfe: float = 0.0
    mae: float = 0.0

    def update_mfe_mae(self, last_price: float) -> None:
        """Update MFE and MAE given the latest market price."""
        if self.side == "LONG":
            # For long, favourable excursion is price minus entry
            self.mfe = max(self.mfe, last_price - self.entry)
            self.mae = max(self.mae, self.entry - last_price)
        else:  # SHORT
            # For short, favourable excursion is entry minus price
            self.mfe = max(self.mfe, self.entry - last_price)
            self.mae = max(self.mae, last_price - self.entry)


class ORBEngine:
    """Compute ORB bounds and momentum/volume confirmations for signals."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = logging.getLogger("EGXIntraday.ORB")

    def analyze(self, df_1m: pd.DataFrame) -> dict:
        """Analyse intraday data and return dictionary with ORB metrics and confirmations."""
        if df_1m.empty:
            return {}
        open_time = parse_time_local(self.cfg["market"]["open_time"])
        # Determine today's date (assume last row belongs to today)
        last_ts = df_1m["Timestamp"].iloc[-1]
        start_day = last_ts.date()
        today_mask = df_1m["Timestamp"].dt.date == start_day
        today = df_1m[today_mask]
        if today.empty:
            return {}
        # Opening range window
        or_start = dt.datetime.combine(start_day, open_time, tzinfo=CAIRO_TZ)
        or_end = or_start + dt.timedelta(minutes=self.cfg["market"]["open_range_minutes"])
        # If late_wake_mode, use the first open_range_minutes from the start
        if self.cfg["strategy"]["late_wake_mode"]:
            # Always compute OR window from market open, not from script start
            pass
        or_window = today[(today["Timestamp"] >= or_start) & (today["Timestamp"] < or_end)]
        if or_window.empty:
            return {}
        orb_high = float(or_window["High"].max())
        orb_low = float(or_window["Low"].min())
        last = today.iloc[-1]
        price = float(last["Close"])
        # Momentum: EMA crossover and RSI
        ema_fast = ema(today["Close"], self.cfg["strategy"]["momentum"]["ema_fast"])
        ema_slow = ema(today["Close"], self.cfg["strategy"]["momentum"]["ema_slow"])
        momentum_up = False
        momentum_down = False
        if len(ema_fast) >= 3 and len(ema_slow) >= 3:
            # Check EMA slope over last 3 periods and crossover
            ema_fast_slope = ema_fast.iloc[-1] > ema_fast.iloc[-3]
            ema_slow_slope = ema_slow.iloc[-1] > ema_slow.iloc[-3]
            ema_cross_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
            ema_cross_down = ema_fast.iloc[-1] < ema_slow.iloc[-1]
            momentum_up = ema_fast_slope and ema_slow_slope and ema_cross_up
            momentum_down = (ema_fast.iloc[-1] < ema_fast.iloc[-3]) and (ema_slow.iloc[-1] < ema_slow.iloc[-3]) and ema_cross_down
        rsi_val = rsi(today["Close"], self.cfg["strategy"]["momentum"]["rsi_len"]).iloc[-1]
        momentum_ok = bool(momentum_up and rsi_val >= self.cfg["strategy"]["momentum"]["rsi_threshold"])
        momentum_down_ok = bool(momentum_down and rsi_val <= (100 - self.cfg["strategy"]["momentum"]["rsi_threshold"]))
        # Volume surge
        vol_ma_len = self.cfg["strategy"]["volume"]["vol_ma_len"]
        surge_factor = self.cfg["strategy"]["volume"]["surge_factor"]
        vol_ma = today["Volume"].rolling(vol_ma_len).mean().iloc[-1] if len(today) >= vol_ma_len else np.nan
        vol_surge = today["Volume"].iloc[-1] > (vol_ma * surge_factor) if not np.isnan(vol_ma) else False
        # Multi‑timeframe confirmation: last candle close > open for long, < open for short
        mtf_ok_long = True
        mtf_ok_short = True
        for tf in self.cfg["strategy"]["multi_timeframes"]:
            rule_map = {"5min": "5min", "15min": "15min", "60min": "60min"}
            r = resample_agg(today, rule_map.get(tf, "5min"))
            if len(r) >= 2:
                mtf_ok_long = mtf_ok_long and (r["Close"].iloc[-1] > r["Open"].iloc[-1])
                mtf_ok_short = mtf_ok_short and (r["Close"].iloc[-1] < r["Open"].iloc[-1])
        return {
            "orb_high": orb_high,
            "orb_low": orb_low,
            "price": price,
            "momentum_up": momentum_ok,
            "momentum_down": momentum_down_ok,
            "vol_surge": vol_surge,
            "mtf_ok_long": mtf_ok_long,
            "mtf_ok_short": mtf_ok_short,
        }


###############################################################################
# Paper Trading Engine
###############################################################################


class PaperTrader:
    """Manage paper positions and apply risk management rules."""

    def __init__(self, cfg: dict, journal: Journal):
        self.cfg = cfg
        self.journal = journal
        # Equity and start equity will be updated each day at EOD
        self.equity_start = cfg["risk"]["daily_start_capital"]
        self.equity = float(self.equity_start)
        self.withdrawn_total = 0.0
        self.max_pnl_today = 0.0
        self.positions: Dict[str, Position] = {}
        self.trades_today = 0
        self.logger = logging.getLogger("EGXIntraday.Trader")

    def reset_day(self) -> None:
        """Reset day counters (trades count, max PnL) at the start of each new trading day."""
        self.max_pnl_today = 0.0
        self.trades_today = 0

    def compute_size(self, price: float, atr: float, confidence: float) -> float:
        """Compute position size (number of shares) based on risk per trade and volatility."""
        # Risk per trade as fraction of equity
        risk_frac = self.cfg["risk"]["position_sizing"]["base_risk_per_trade"] / 100.0
        risk_budget = self.equity * risk_frac
        # Stop distance approximates the risk per share
        # Use ATR * primary_buffer multiplier or a minimum tick distance
        primary_mult = self.cfg["strategy"]["primary_buffer_atr_mult"]
        stop_dist = max(atr * primary_mult, price * 0.003)
        if stop_dist <= 0:
            return 0.0
        qty = risk_budget / stop_dist
        # Scale size based on volatility (cap extremes) and confidence
        if self.cfg["risk"]["position_sizing"]["confidence_scale"]:
            qty *= (0.5 + confidence)  # 0.5–1.5x scaling
        # Limit oversizing in quiet markets
        volatility_cap = self.cfg["risk"]["position_sizing"]["volatility_scale_cap"]
        if atr > 0:
            # Lower volatility → larger size; apply cap multiplier
            # Use ratio to typical ATR (here we normalise by price)
            normalised_atr = atr / max(price, 1e-6)
            scaling_factor = 1.0 / max(normalised_atr, 1e-6)
            scaling_factor = min(scaling_factor, volatility_cap)
            qty *= scaling_factor
        return max(qty, 0.0)

    def apply_transaction_costs(self, price: float, qty: float) -> float:
        """Calculate transaction costs for entry or exit based on price and quantity."""
        cost_cfg = self.cfg["risk"].get("transaction_cost", {})
        commission = cost_cfg.get("commission_rate", 0.0)
        stamp_tax = cost_cfg.get("stamp_tax_rate", 0.0)
        return price * qty * (commission + stamp_tax)

    def on_signal(self, sig: Signal, atr: float) -> None:
        """Execute a paper trade when a signal triggers (new or add‑on).

        If a signal cannot be executed due to daily trade limits or zero sizing,
        log the missed opportunity along with a hypothetical position size and ORB bounds.
        """
        # Enforce max trades per day
        if self.trades_today >= self.cfg["risk"]["max_trades_per_day"]:
            # Log missed opportunity due to trade limit
            hypothetical_qty = self.compute_size(sig.price, atr, sig.confidence)
            self.journal.log_missed_opportunity(
                ts=sig.ts.isoformat(), symbol=sig.symbol, side=sig.side,
                price=sig.price, qty=hypothetical_qty, orb_high=sig.orb_high,
                orb_low=sig.orb_low, reason="Max trades per day"
            )
            self.logger.info("Max trades per day reached; skipping signal")
            return
        # Check if symbol is already in positions (potential add-on)
        if sig.symbol in self.positions:
            pos = self.positions[sig.symbol]
            # Only allow additional entries if configured and not exceeded
            if self.cfg["strategy"]["allow_multi_entry"] and pos.adds_done < self.cfg["strategy"]["max_adds"]:
                add_qty = 0.5 * self.compute_size(sig.price, atr, sig.confidence)
                if add_qty <= 0:
                    # Log missed opportunity for add-on with zero size
                    self.journal.log_missed_opportunity(
                        ts=sig.ts.isoformat(), symbol=sig.symbol, side=sig.side,
                        price=sig.price, qty=0.0, orb_high=sig.orb_high,
                        orb_low=sig.orb_low, reason="Zero add-on qty"
                    )
                    return
                # Execute add-on
                pos.qty += add_qty
                pos.adds_done += 1
                self.trades_today += 1
                self.journal.log_trade(
                    ts_open=sig.ts.isoformat(), ts_close=None, symbol=sig.symbol,
                    side=f"ADD_{pos.side}", entry=sig.price, exit=None, qty=add_qty,
                    pnl=0.0, fees=self.apply_transaction_costs(sig.price, add_qty), mfe=pos.mfe, mae=pos.mae,
                    adds=pos.adds_done, notes="Additional entry"
                )
                return
            else:
                # Cannot add further entries
                self.journal.log_missed_opportunity(
                    ts=sig.ts.isoformat(), symbol=sig.symbol, side=sig.side,
                    price=sig.price, qty=0.0, orb_high=sig.orb_high,
                    orb_low=sig.orb_low, reason="Max adds reached or add-on disabled"
                )
                return
        # New position
        side = "LONG" if sig.side == "BUY" else "SHORT"
        # Compute buffers
        pb = atr * self.cfg["strategy"]["primary_buffer_atr_mult"]
        rb = atr * self.cfg["strategy"]["reserve_buffer_atr_mult"]
        qty = self.compute_size(sig.price, atr, sig.confidence)
        if qty <= 0:
            # Log missed opportunity for zero quantity
            self.journal.log_missed_opportunity(
                ts=sig.ts.isoformat(), symbol=sig.symbol, side=sig.side,
                price=sig.price, qty=0.0, orb_high=sig.orb_high,
                orb_low=sig.orb_low, reason="Zero quantity"
            )
            return
        # Create new position
        pos = Position(
            symbol=sig.symbol,
            side=side,
            entry=sig.price,
            qty=qty,
            ts_open=sig.ts,
            orb_high=sig.orb_high,
            orb_low=sig.orb_low,
            buffers={"primary": pb, "reserve": rb},
            trail_stop=None,
        )
        # Initialize trailing stop
        if side == "LONG":
            pos.trail_stop = sig.price - self.cfg["strategy"]["trailing_atr_mult"] * atr
        else:
            pos.trail_stop = sig.price + self.cfg["strategy"]["trailing_atr_mult"] * atr
        self.positions[sig.symbol] = pos
        self.trades_today += 1
        # Log new trade
        fees = self.apply_transaction_costs(sig.price, qty)
        self.journal.log_trade(
            ts_open=sig.ts.isoformat(), ts_close=None, symbol=sig.symbol,
            side=side, entry=sig.price, exit=None, qty=qty,
            pnl=0.0, fees=fees, mfe=0.0, mae=0.0, adds=0, notes="New position"
        )

    def update_position(self, symbol: str, last_price: float, atr: float, ts: dt.datetime) -> None:
        """Update trailing stop, MFE/MAE and exit logic for an open position."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        # Update MFE and MAE
        pos.update_mfe_mae(last_price)
        # Update trailing stop only in favourable direction
        if pos.side == "LONG":
            new_trail = last_price - self.cfg["strategy"]["trailing_atr_mult"] * atr
            if new_trail > (pos.trail_stop or -float("inf")):
                pos.trail_stop = new_trail
        else:
            new_trail = last_price + self.cfg["strategy"]["trailing_atr_mult"] * atr
            if pos.trail_stop is None or new_trail < pos.trail_stop:
                pos.trail_stop = new_trail
        # Determine risk (distance to initial stop)
        if pos.side == "LONG":
            risk = max(pos.entry - pos.orb_low, pos.buffers["primary"])
            take_profit = pos.entry + self.cfg["strategy"]["take_profit_rr"] * risk
        else:
            risk = max(pos.orb_high - pos.entry, pos.buffers["primary"])
            take_profit = pos.entry - self.cfg["strategy"]["take_profit_rr"] * risk
        exit_price: Optional[float] = None
        reason = ""
        # Check trailing stop
        if pos.side == "LONG" and last_price <= (pos.trail_stop or -float("inf")):
            exit_price = pos.trail_stop
            reason = "Trailing stop hit"
        elif pos.side == "SHORT" and last_price >= (pos.trail_stop or float("inf")):
            exit_price = pos.trail_stop
            reason = "Trailing stop hit"
        # Check take profit
        if exit_price is None:
            if pos.side == "LONG" and last_price >= take_profit:
                exit_price = take_profit
                reason = "Take profit"
            elif pos.side == "SHORT" and last_price <= take_profit:
                exit_price = take_profit
                reason = "Take profit"
        # Only close when conditions met
        if exit_price is not None:
            pnl = 0.0
            if pos.side == "LONG":
                pnl = (exit_price - pos.entry) * pos.qty
            else:
                pnl = (pos.entry - exit_price) * pos.qty
            # Deduct exit transaction costs
            fees = self.apply_transaction_costs(exit_price, pos.qty)
            net_pnl = pnl - fees
            self.equity += net_pnl
            self.max_pnl_today = max(self.max_pnl_today, self.equity - self.equity_start)
            # Remove position
            del self.positions[symbol]
            # Update trades table entry with exit info
            self.journal.log_trade(
                ts_open=pos.ts_open.isoformat(), ts_close=ts.isoformat(), symbol=pos.symbol,
                side=f"EXIT_{pos.side}", entry=pos.entry, exit=exit_price, qty=pos.qty,
                pnl=net_pnl, fees=fees, mfe=pos.mfe, mae=pos.mae, adds=pos.adds_done, notes=reason
            )

    def risk_brakes(self) -> Tuple[bool, str]:
        """Check for capital protection, daily loss limits or profit locks."""
        # Calculate daily PnL (equity minus start equity)
        pnl_today = self.equity - self.equity_start
        eq_ratio = self.equity / self.equity_start if self.equity_start else 1.0
        # Capital protection: stop trading if equity falls below threshold
        if eq_ratio < self.cfg["risk"]["capital_protection_buffer"]:
            return True, "Capital protection buffer breached"
        # Daily loss limit
        if pnl_today <= -self.cfg["risk"]["daily_loss_limit"] * self.equity_start:
            return True, "Daily loss limit reached"
        # Profit lock: if we have significant gains and giveback exceeds threshold
        lock_thr = self.cfg["risk"]["profit_lock_trigger"] * self.equity_start
        giveback_allow = self.cfg["risk"]["profit_lock_giveback"] * lock_thr
        if self.max_pnl_today >= lock_thr and (self.max_pnl_today - pnl_today) > giveback_allow:
            return True, "Profit lock activated"
        return False, ""

    def eod_settle(self) -> Tuple[str, float, float]:
        """Perform end‑of‑day settlement: withdraw profits and update starting equity."""
        pnl_today = self.equity - self.equity_start
        withdraw_amt = 0.0
        note = ""
        if pnl_today > 0:
            mode = self.cfg["risk"].get("withdrawal_mode", "fixed")
            if mode == "fixed":
                max_withdraw = self.cfg["risk"].get("daily_withdrawal", 0.0)
                withdraw_amt = min(max_withdraw, pnl_today)
            elif mode == "percent":
                pct = self.cfg["risk"].get("withdrawal_percent", 0.0)
                withdraw_amt = pnl_today * pct
            elif mode == "hybrid":
                # Hybrid: withdraw fixed first then a percent of remaining profit
                fixed_amt = self.cfg["risk"].get("daily_withdrawal", 0.0)
                pct = self.cfg["risk"].get("withdrawal_percent", 0.0)
                withdraw_amt = min(fixed_amt, pnl_today)
                remaining = pnl_today - withdraw_amt
                if remaining > 0:
                    withdraw_amt += remaining * pct
            # Apply withdrawal
            self.withdrawn_total += withdraw_amt
            self.equity -= withdraw_amt
            note = f"💰 Withdrew {withdraw_amt:.2f} EGP from profits"
        else:
            note = "No withdrawal (loss day)"
        # Set next day equity baseline
        self.equity_start = self.equity
        return note, pnl_today, self.equity

    def unrealized_pnl(self, price_map: Dict[str, float]) -> Tuple[float, List[Tuple[str, float]]]:
        """Calculate unrealized PnL for all open positions given current prices.

        Returns the total unrealized PnL and a list of tuples (symbol, pnl).
        """
        total = 0.0
        details: List[Tuple[str, float]] = []
        for sym, pos in list(self.positions.items()):
            if sym not in price_map or price_map[sym] is None:
                continue
            last_price = price_map[sym]
            if pos.side == "LONG":
                pnl = (last_price - pos.entry) * pos.qty
            else:
                pnl = (pos.entry - last_price) * pos.qty
            # Deduct exit costs as estimate (exit may occur next day)
            fees = self.apply_transaction_costs(last_price, pos.qty)
            net = pnl - fees
            total += net
            details.append((sym, net))
        return total, details


###############################################################################
# Charts and Visualisations
###############################################################################


def render_chart_png(df: pd.DataFrame, title: str = "Chart") -> bytes:
    """Render a candlestick chart with volume and return bytes of a PNG."""
    if mpf is None or df.empty:
        return b""
    # Use last 200 rows for clarity
    dfp = df.copy().tail(200).set_index("Timestamp")[["Open", "High", "Low", "Close", "Volume"]]
    buf = io.BytesIO()
    try:
        mpf.plot(dfp, type="candle", volume=True, style="yahoo", title=title, savefig=buf)
        return buf.getvalue()
    except Exception:
        return b""


###############################################################################
# Watchlist and Heatmap
###############################################################################


def rank_watchlist(data_map: Dict[str, pd.DataFrame], cfg: dict) -> List[str]:
    """Rank symbols based on gap, volume surge and momentum slope."""
    scores = []
    for sym, df in data_map.items():
        if df.empty:
            continue
        today = df[df["Timestamp"].dt.date == now_cairo().date()]
        if len(today) < 2:
            continue
        o = float(today["Open"].iloc[0])
        c = float(today["Close"].iloc[-1])
        gap = (c - o) / max(o, 1e-6)
        vol_ma = today["Volume"].rolling(20).mean().iloc[-1]
        vol_surge = today["Volume"].iloc[-1] / max(vol_ma, 1e-6) if not np.isnan(vol_ma) else 0.0
        ef = ema(today["Close"], cfg["strategy"]["momentum"]["ema_fast"])
        es = ema(today["Close"], cfg["strategy"]["momentum"]["ema_slow"])
        if len(ef) >= 1 and len(es) >= 1:
            mom = (ef.iloc[-1] - es.iloc[-1]) / max(es.iloc[-1], 1e-6)
        else:
            mom = 0.0
        w = cfg["watchlist"]["rank_features"]
        score = (
            w["gap_weight"] * gap
            + w["vol_surge_weight"] * min(vol_surge, 5.0)
            + w["momentum_weight"] * mom
        )
        scores.append((sym, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[: cfg["watchlist"]["top_n"]]]


def sector_heatmap(data_map: Dict[str, pd.DataFrame], sectors: Dict[str, str]) -> Dict[str, float]:
    """Aggregate intraday performance by sector."""
    agg: Dict[str, List[float]] = {}
    for sym, df in data_map.items():
        if df.empty:
            continue
        sec = sectors.get(sym, "Other")
        today = df[df["Timestamp"].dt.date == now_cairo().date()]
        if len(today) < 2:
            continue
        perf = (today["Close"].iloc[-1] - today["Open"].iloc[0]) / max(today["Open"].iloc[0], 1e-6)
        agg.setdefault(sec, []).append(perf)
    return {k: float(np.mean(v)) for k, v in agg.items() if v}


###############################################################################
# News and Sentiment (Optional)
###############################################################################


def fetch_rss_items(urls: List[str], max_items: int = 20) -> List[Dict[str, str]]:
    """Fetch RSS feed items from provided URLs.  Returns list of dicts."""
    items: List[Dict[str, str]] = []
    for url in urls:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.content)
            for it in root.iter("item"):
                title = it.findtext("title") or ""
                desc = it.findtext("description") or ""
                pub = it.findtext("pubDate") or ""
                items.append({"title": title, "desc": desc, "pub": pub})
        except Exception:
            continue
    return items[:max_items]


def simple_sentiment(text: str) -> str:
    """Naive sentiment classifier based on keywords."""
    text_l = text.lower()
    pos_words = ["beat", "surge", "growth", "record", "strong", "increase", "upgrade", "positive", "profit"]
    neg_words = ["miss", "drop", "fall", "decline", "downgrade", "negative", "loss", "warning"]
    score = sum(w in text_l for w in pos_words) - sum(w in text_l for w in neg_words)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


###############################################################################
# Telegram Poller
###############################################################################


class TelegramPoller(threading.Thread):
    """Background thread to poll Telegram for callback queries and log user actions."""

    def __init__(self, bot: Telegram, journal: Journal):
        super().__init__(daemon=True)
        self.bot = bot
        self.journal = journal
        self.offset: Optional[int] = None
        self.logger = logging.getLogger("EGXIntraday.TelegramPoller")

    def run(self) -> None:
        while True:
            try:
                params = {"timeout": 30}
                if self.offset is not None:
                    params["offset"] = self.offset
                r = requests.get(f"{self.bot.base}/getUpdates", params=params, timeout=40)
                if r.status_code != 200:
                    time.sleep(5)
                    continue
                updates = r.json().get("result", [])
                for upd in updates:
                    self.offset = upd["update_id"] + 1
                    cb = upd.get("callback_query")
                    if not cb:
                        continue
                    data = cb.get("data", "")
                    ts = now_cairo().isoformat()
                    if data.startswith("buy:"):
                        sym = data.split(":")[1]
                        self.journal.log_miss(ts=ts, symbol=sym, reason="User clicked Buy")
                    elif data.startswith("skip:"):
                        sym = data.split(":")[1]
                        self.journal.log_miss(ts=ts, symbol=sym, reason="User skipped")
                    elif data.startswith("adjust:"):
                        sym = data.split(":")[1]
                        self.journal.log_miss(ts=ts, symbol=sym, reason="User adjust buffers")
            except Exception:
                time.sleep(5)


###############################################################################
# Report Generation and Archiving
###############################################################################


def generate_reports(db_path: str, tg: Telegram, cfg: dict, mode: str = "weekly") -> None:
    """Generate and send weekly or monthly KPI reports, archive CSV and PNG files."""
    if plt is None:
        tg.send_message(f"📊 {mode.capitalize()} report: Matplotlib not available")
        return
    con = sqlite3.connect(db_path)
    trades = pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["ts_open", "ts_close"])
    con.close()
    if trades.empty:
        tg.send_message(f"📊 {mode.capitalize()} report: No trades logged.")
        return
    trades["date"] = trades["ts_open"].dt.tz_localize("UTC").dt.tz_convert(CAIRO_TZ).dt.date
    trades["week"] = trades["ts_open"].dt.to_period("W")
    trades["month"] = trades["ts_open"].dt.to_period("M")
    if mode == "weekly":
        grp = trades.groupby("week")["pnl"].sum()
        label = str(grp.index[-1])
    else:
        grp = trades.groupby("month")["pnl"].sum()
        label = str(grp.index[-1])
    total_pnl = grp.iloc[-1]
    msg = (
        f"📊 <b>{mode.capitalize()} KPI</b>\n"
        f"Period: {label}\n"
        f"PnL: {total_pnl:.2f}\n"
        f"Trades: {len(trades)}"
    )
    tg.send_message(msg)
    # Prepare directory
    out_dir = os.path.join(cfg["journal"]["report_out_dir"], mode)
    os.makedirs(out_dir, exist_ok=True)
    # Save CSV
    csv_path = os.path.join(out_dir, f"{mode}_{label}.csv")
    trades.to_csv(csv_path, index=False)
    # PnL bar chart
    fig, ax = plt.subplots()
    grp.plot(kind="bar", ax=ax, title=f"{mode.capitalize()} PnL")
    ax.set_xlabel(mode.capitalize())
    ax.set_ylabel("PnL")
    png_path = os.path.join(out_dir, f"{mode}_pnl_{label}.png")
    fig.savefig(png_path, bbox_inches="tight")
    tg.send_plot(fig, caption=f"{mode.capitalize()} PnL History")
    plt.close(fig)
    # Buffer distribution chart
    bufs = trades["buffers"].dropna().apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    primary_bufs = [b.get("primary", 0) for b in bufs if isinstance(b, dict)]
    if primary_bufs:
        fig2, ax2 = plt.subplots()
        ax2.hist(primary_bufs, bins=20)
        ax2.set_title("Primary buffer distribution")
        ax2.set_xlabel("ATR * Multiplier")
        ax2.set_ylabel("Count")
        buf_png = os.path.join(out_dir, f"{mode}_buffers_{label}.png")
        fig2.savefig(buf_png, bbox_inches="tight")
        tg.send_plot(fig2, caption="Buffer distribution")
        plt.close(fig2)
    # Auto‑compress archives older than 90 days
    try:
        import zipfile
        import glob
        cutoff = dt.date.today() - dt.timedelta(days=90)
        files = glob.glob(os.path.join(out_dir, "*.csv")) + glob.glob(os.path.join(out_dir, "*.png"))
        for f in files:
            fname = os.path.basename(f)
            # Extract date component from filename
            parts = fname.split("_")
            if len(parts) < 2:
                continue
            date_part = parts[-1].split(".")[0]
            # Parse weekly (YYYY-Wxx) or monthly (YYYY-MM)
            file_date: Optional[dt.date] = None
            try:
                if "W" in date_part:
                    yr, wk = date_part.split("-W")
                    file_date = dt.date.fromisocalendar(int(yr), int(wk), 1)
                else:
                    yr, mo = date_part.split("-")
                    file_date = dt.date(int(yr), int(mo), 1)
            except Exception:
                file_date = None
            if file_date and file_date < cutoff:
                zip_path = os.path.join(out_dir, f"archive_{mode}.zip")
                with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(f, arcname=fname)
                os.remove(f)
    except Exception:
        pass


###############################################################################
# EGX Assistant
###############################################################################


class EGXAssistant:
    """Main orchestrator for fetching data, detecting signals and trading."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.router = DataRouter(cfg)
        self.tg = Telegram(cfg["alerts"]["telegram_bot_token"], cfg["alerts"]["telegram_chat_id"])
        self.journal = Journal(cfg["journal"]["db_path"])
        self.engine = ORBEngine(cfg)
        self.trader = PaperTrader(cfg, self.journal)
        self.ml_scorer = SignalScorer(cfg["journal"]["db_path"])
        # For weekly/monthly reporting schedule
        self.last_week: Optional[int] = None
        self.last_month: Optional[int] = None
        self.logger = logging.getLogger("EGXIntraday.Assistant")

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch intraday data for all symbols via the router using threads."""
        m: Dict[str, pd.DataFrame] = {}
        syms = self.cfg.get("symbols", [])
        interval = self.cfg["data"].get("interval", "1m")
        lookback = self.cfg["data"].get("lookback_minutes", 240)
        # Limit number of workers to avoid overwhelming providers
        max_workers = min(len(syms), 4) if syms else 0
        if max_workers <= 1:
            # Fallback to sequential fetch when only one symbol or no concurrency
            for sym in syms:
                df = self.router.fetch(sym, interval, lookback)
                m[sym] = ensure_cols(df) if df is not None else pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
            return m
        # Use ThreadPoolExecutor for concurrent symbol fetches
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {
                executor.submit(self.router.fetch, sym, interval, lookback): sym
                for sym in syms
            }
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    df = future.result()
                except Exception as exc:
                    logging.getLogger("EGXIntraday").warning(f"Error fetching {sym}: {exc}")
                    df = pd.DataFrame()
                m[sym] = ensure_cols(df) if df is not None else pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        return m

    def generate_signal(self, sym: str, df: pd.DataFrame) -> Optional[Signal]:
        """Generate a buy or sell signal based on ORB analysis and ML scoring."""
        if df.empty:
            return None
        check = self.engine.analyze(df)
        if not check:
            return None
        price = check["price"]
        orb_high = check["orb_high"]
        orb_low = check["orb_low"]
        # Determine long (break above orb_high) or short (break below orb_low)
        last = df.iloc[-1]
        crossed_long = last["High"] >= orb_high and last["Close"] > orb_high
        crossed_short = last["Low"] <= orb_low and last["Close"] < orb_low
        sig_side: Optional[str] = None
        # Confirmations
        confirms_long = [check["momentum_up"], check["vol_surge"], check["mtf_ok_long"]]
        confirms_short = [check["momentum_down"], check["vol_surge"], check["mtf_ok_short"]]
        # Confidence computation; treat long and short separately
        conf_long = sum(confirms_long) / 3.0
        conf_short = sum(confirms_short) / 3.0
        if crossed_long and conf_long >= self.cfg["strategy"]["min_confidence_for_alert"]:
            sig_side = "BUY"
            conf = conf_long
        elif self.cfg["short"]["allow_short"] and crossed_short and conf_short >= self.cfg["strategy"]["min_confidence_for_alert"]:
            # Only allow short on designated symbols if short_list is non‑empty
            if self.cfg["short"]["short_list"] and sym not in self.cfg["short"]["short_list"]:
                sig_side = None
            else:
                sig_side = "SELL"
                conf = conf_short
        else:
            return None
        # Apply ML scorer on features
        # Feature set: risk proxy (entry minus ORB bound), adds = 0, entry price, qty=1
        risk_proxy = (price - orb_low) if sig_side == "BUY" else (orb_high - price)
        ml_features = {"risk": risk_proxy, "adds": 0, "entry": price, "qty": 1}
        ml_score = self.ml_scorer.score(ml_features)
        confidence = 0.5 * conf + 0.5 * ml_score
        provider = str(df.get("Provider", pd.Series(["unknown"])).iloc[-1]) if "Provider" in df.columns else "unknown"
        # Generate chart if requested
        chart_bytes = b""
        if self.cfg["alerts"]["send_charts"]:
            chart_bytes = render_chart_png(df, f"{sym} ORB")
        return Signal(
            symbol=sym,
            ts=now_cairo(),
            side=sig_side,
            reason=f"ORB breakout {'long' if sig_side=='BUY' else 'short'}",
            confidence=confidence,
            price=price,
            orb_high=orb_high,
            orb_low=orb_low,
            provider=provider,
            chart=chart_bytes if chart_bytes else None,
        )

    def send_alert(self, sig: Signal) -> None:
        """Send a signal alert via Telegram and log it."""
        text = (
            f"<b>{sig.symbol}</b> • <i>{'LONG' if sig.side=='BUY' else 'SHORT'}</i>\n"
            f"Price: <b>{sig.price:.3f}</b>\n"
            f"ORB(H/L): {sig.orb_high:.3f} / {sig.orb_low:.3f}\n"
            f"Conf: <b>{sig.confidence:.2f}</b> • Provider: {sig.provider}\n"
            f"Reason: {sig.reason}\n"
            f"<i>Paper‑trade only • No execution</i>"
        )
        buttons = [
            [
                {"text": "Buy (Paper)", "callback_data": f"buy:{sig.symbol}:{sig.price:.4f}"},
                {"text": "Skip", "callback_data": f"skip:{sig.symbol}"},
                {"text": "Adjust", "callback_data": f"adjust:{sig.symbol}"},
            ]
        ]
        if sig.chart:
            # send as photo if chart available
            self.tg.send_photo(caption=text, image_bytes=sig.chart)
            chart_sent = True
        else:
            self.tg.send_message(text, buttons=buttons)
            chart_sent = False
        # Log alert
        self.journal.log_alert(
            ts=sig.ts.isoformat(), symbol=sig.symbol, side=sig.side,
            reason=sig.reason, confidence=sig.confidence, price=sig.price,
            orb_high=sig.orb_high, orb_low=sig.orb_low, provider=sig.provider,
            chart_sent=chart_sent,
        )

    def run_once(self) -> None:
        """One iteration: fetch data, generate signals, manage positions and risk."""
        # Risk brakes: if triggered, stop trading and close positions
        stop, why = self.trader.risk_brakes()
        if stop:
            self.tg.send_message(f"⛔️ Trading paused: {why}")
            return
        # Fetch data
        data_map = self.fetch_all()
        # Watchlist and heatmap (send daily once or per run?  We'll send on each run)
        wl = rank_watchlist(data_map, self.cfg)
        heat = sector_heatmap(data_map, self.cfg.get("sectors", {}))
        if wl:
            heat_txt = " | ".join(f"{k}:{v:+.2%}" for k, v in sorted(heat.items(), key=lambda x: x[1], reverse=True))
            self.tg.send_message(
                f"Watchlist: <b>{', '.join(wl)}</b>\nSectors: {heat_txt}"
            )
        # News pulse (optional)
        if self.cfg["data"]["rss_feeds"]:
            items = fetch_rss_items(self.cfg["data"]["rss_feeds"], self.cfg["data"]["news_max_items"])
            if items:
                positive = [it for it in items if simple_sentiment(it["title"]) == "positive"][:3]
                negative = [it for it in items if simple_sentiment(it["title"]) == "negative"][:3]
                msg_parts = []
                if positive:
                    msg_parts.append("✅ Pos: " + " | ".join(p["title"][:80] for p in positive))
                if negative:
                    msg_parts.append("⚠️ Neg: " + " | ".join(n["title"][:80] for n in negative))
                if msg_parts:
                    self.tg.send_message("📰 <b>News pulse</b>\n" + "\n".join(msg_parts))
        # Generate signals and update positions
        for sym, df in data_map.items():
            if df.empty:
                self.journal.log_miss(ts=now_cairo().isoformat(), symbol=sym, reason="No data")
                continue
            sig = self.generate_signal(sym, df)
            if sig:
                # Compute ATR for sizing and buffer definitions
                atr = atr_like(df, 14)
                # Send alert
                self.send_alert(sig)
                # Execute paper trade
                self.trader.on_signal(sig, atr)
            # Update open positions with latest price
            last_price = float(df["Close"].iloc[-1]) if not df.empty else np.nan
            atr_val = atr_like(df, 14)
            self.trader.update_position(sym, last_price, atr_val, now_cairo())

    def eod_reports(self) -> None:
        """Send end‑of‑day KPIs and handle settlement/archiving."""
        day = now_cairo().date()
        rep = self.journal.daily_report(day)
        txt = (
            f"📊 <b>Daily KPIs ({day})</b>\n"
            f"Hit rate: {rep['hit_rate'] if rep['hit_rate'] is not None else 'n/a'}\n"
            f"Net PnL: {rep['net_pnl']:.2f}\n"
            f"Trades: {rep['num_trades']}\n"
            f"Best: {rep['best_symbol']} | Worst: {rep['worst_symbol']}"
        )
        self.tg.send_message(txt)

        # Compute unrealized PnL for open positions using latest closing prices
        # Fetch latest close for each symbol (intraday fetch with small lookback)
        price_map: Dict[str, float] = {}
        for sym in self.cfg["symbols"]:
            df = self.router.fetch(sym, self.cfg["data"]["interval"], self.cfg["data"].get("lookback_minutes", 30))
            if df is not None and not df.empty:
                df = ensure_cols(df)
                price_map[sym] = float(df.iloc[-1]["Close"])
        unreal_total, unreal_details = self.trader.unrealized_pnl(price_map)

        # Update missed opportunities with potential PnL for this day
        missed_df = self.journal.fetch_missed_opps(day)
        missed_summary = ""
        if not missed_df.empty:
            # Compute potential PnL based on closing prices
            for idx, row in missed_df.iterrows():
                sym = row["symbol"]
                side = row["side"]
                price = float(row["price"])
                qty = float(row["qty"])
                orb_low = row.get("orb_low")
                orb_high = row.get("orb_high")
                # Use closing price from price_map if available
                close_price = price_map.get(sym)
                if close_price is None:
                    continue
                # Compute potential PnL for missed trade
                if side in ("BUY", "LONG"):
                    pnl = (close_price - price) * qty
                else:
                    pnl = (price - close_price) * qty
                # Deduct estimated transaction costs for hypothetical exit
                fees = self.trader.apply_transaction_costs(close_price, qty)
                potential_pnl = pnl - fees
                # Update missed record if not already updated
                if pd.isna(row["potential_pnl"]):
                    self.journal.update_missed_pnl(int(row["id"]), potential_pnl, now_cairo())
            # Reload updated missed opportunities for summary
            missed_df = self.journal.fetch_missed_opps(day)
            pos_missed = missed_df[missed_df["potential_pnl"] > 0]
            neg_missed = missed_df[missed_df["potential_pnl"] < 0]
            missed_summary = (
                f"Missed signals: {len(missed_df)}\n"
                f"Profitable misses: {len(pos_missed)} for +{pos_missed['potential_pnl'].sum():.2f}\n"
                f"Losing misses: {len(neg_missed)} for {neg_missed['potential_pnl'].sum():.2f}"
            )
            # Summarise reasons
            reason_counts = missed_df["reason"].value_counts().to_dict()
            reasons_str = ", ".join(f"{k}:{v}" for k, v in reason_counts.items())
            if reasons_str:
                missed_summary += f"\nReasons: {reasons_str}"

        # Apply end‑of‑day settlement (withdraw profits and update equity baseline)
        note, pnl_today, new_equity = self.trader.eod_settle()
        summary_lines = [
            f"{note}",
            f"PnL today: {pnl_today:.2f}",
            f"Reinvested capital for tomorrow: {new_equity:.2f}",
            f"Unrealized PnL: {unreal_total:.2f}",
            f"Total withdrawn so far: {self.trader.withdrawn_total:.2f}",
        ]
        if missed_summary:
            summary_lines.append(missed_summary)
        self.tg.send_message("\n".join(summary_lines))

        # Write out daily report CSV snapshot (for backup)
        con = sqlite3.connect(self.cfg["journal"]["db_path"])
        df_tr = pd.read_sql_query("SELECT * FROM trades", con)
        con.close()
        outp_dir = self.cfg["journal"]["report_out_dir"]
        os.makedirs(outp_dir, exist_ok=True)
        outp_path = os.path.join(outp_dir, f"kpi_{day}.csv")
        df_tr.to_csv(outp_path, index=False)

        # Retrain ML model after updating missed opportunities
        try:
            self.ml_scorer.train()
        except Exception:
            pass

    def schedule_reports(self) -> None:
        """Handle weekly and monthly report scheduling."""
        now = now_cairo()
        # Weekly: every Friday after close
        if now.weekday() == 4 and self.last_week != now.isocalendar()[1]:
            generate_reports(self.cfg["journal"]["db_path"], self.tg, self.cfg, mode="weekly")
            self.last_week = now.isocalendar()[1]
        # Monthly: last trading day of month after close
        tomorrow = now + dt.timedelta(days=1)
        if tomorrow.month != now.month and self.last_month != now.month:
            generate_reports(self.cfg["journal"]["db_path"], self.tg, self.cfg, mode="monthly")
            self.last_month = now.month


###############################################################################
# Main Loop
###############################################################################


def main() -> None:
    """Entry point: initialise assistant, poller and loop during market hours."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    cfg = load_config()
    assistant = EGXAssistant(cfg)
    # Start Telegram poller if token is provided
    poller = None
    if cfg["alerts"].get("telegram_bot_token") and cfg["alerts"].get("telegram_chat_id"):
        poller = TelegramPoller(assistant.tg, assistant.journal)
        poller.start()
    # Main loop
    try:
        last_day = now_cairo().date()
        while True:
            now = now_cairo()
            if between_market_hours(now, cfg):
                # Check if new day (just after midnight or weekend)
                if now.date() != last_day:
                    assistant.trader.reset_day()
                    last_day = now.date()
                assistant.run_once()
                time.sleep(60)
            else:
                # Outside trading hours: send EOD report if just after close
                close_time = parse_time_local(cfg["market"]["close_time"])
                if now.time() > close_time and now.minute in (0, 1, 2, 3, 4):
                    assistant.eod_reports()
                # Schedule weekly/monthly reports after EOD
                assistant.schedule_reports()
                time.sleep(30)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()