"""Interview-ready Monte Carlo DCF valuation engine.

Design goals
------------
1. Keep data access separate from valuation logic.
2. Make every analyst judgement configurable from the CLI or a JSON file.
3. Reconcile cash-flow-based and bottom-up FCFF instead of trusting one row.
4. Normalize temporary accounting noise before projecting a perpetuity.
5. Fail with an actionable message when source data is incomplete.

This project is for education and research. It is not investment advice.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Protocol, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


class ValuationError(RuntimeError):
    """Base exception for a controlled valuation failure."""


class DataUnavailableError(ValuationError):
    """Raised when a required source item is unavailable."""


class ModelInputError(ValuationError):
    """Raised when an analyst assumption violates a model constraint."""


@dataclass(frozen=True, slots=True)
class ValuationConfig:
    """All analyst-controlled assumptions.

    Values use decimal rates: 0.05 means 5%. Every field below can be
    overridden by a JSON configuration file or a command-line option.
    """

    ticker: str

    # ==================== USER-ADJUSTABLE: VALUATION ASSUMPTIONS ====================
    explicit_growth_rate: float | None = None  # None = historical auto-anchor
    terminal_growth_rate: float = 0.025
    forecast_years: int = 5
    equity_risk_premium: float = 0.05
    risk_free_rate_override: float | None = None
    risk_free_rate_fallback: float = 0.04
    beta_override: float | None = None
    beta_fallback: float = 1.0
    wacc_override: float | None = None
    wacc_floor: float = 0.075
    minimum_equity_spread: float = 0.02
    target_debt_weight: float | None = None

    # ==================== USER-ADJUSTABLE: FCFF METHODOLOGY ====================
    # normalized: recent cash-flow FCFF median (recommended for mature firms)
    # latest: latest cash-flow FCFF
    # bottom_up: EBIT-based FCFF
    fcff_method: str = "normalized"
    fcff_override: float | None = None  # raw currency amount, not billions
    normalization_years: int = 3
    fcff_divergence_warning: float = 0.20
    latest_to_normalized_warning: float = 0.25

    # ==================== USER-ADJUSTABLE: AUTOMATIC GROWTH ESTIMATION ====================
    auto_growth_revenue_weight: float = 0.70
    auto_growth_fcff_weight: float = 0.30
    auto_growth_fallback: float = 0.03
    auto_growth_bounds: tuple[float, float] = (-0.02, 0.08)

    # ==================== USER-ADJUSTABLE: TAX AND MARKET DATA ====================
    default_tax_rate: float = 0.21
    maximum_tax_rate: float = 0.35
    market_symbol: str = "^GSPC"
    risk_free_symbol: str = "^TNX"
    risk_period: str = "5y"
    risk_interval: str = "1mo"

    # ==================== USER-ADJUSTABLE: MONTE CARLO ASSUMPTIONS ====================
    simulations: int = 10_000
    random_seed: int = 42
    wacc_std: float = 0.01
    explicit_growth_std: float = 0.02
    terminal_growth_std: float = 0.003
    min_terminal_spread: float = 0.01
    growth_bounds: tuple[float, float] = (-0.20, 0.40)
    terminal_growth_bounds: tuple[float, float] = (-0.01, 0.05)
    wacc_bounds: tuple[float, float] = (0.03, 0.30)
    minimum_valid_fraction: float = 0.80

    # ==================== USER-ADJUSTABLE: DATA ACCESS ====================
    # auto = SEC statements when sec_user_agent is provided, otherwise Yahoo.
    data_source: str = "auto"
    sec_user_agent: str | None = None  # e.g. "Your Name your.email@example.com"
    request_retries: int = 2
    retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", self.ticker.strip().upper())
        object.__setattr__(self, "market_symbol", self.market_symbol.strip().upper())
        object.__setattr__(
            self, "risk_free_symbol", self.risk_free_symbol.strip().upper()
        )
        object.__setattr__(self, "fcff_method", self.fcff_method.strip().lower())
        object.__setattr__(self, "data_source", self.data_source.strip().lower())

        if not self.ticker:
            raise ModelInputError("ticker must not be empty")
        if self.data_source not in {"auto", "yahoo", "sec"}:
            raise ModelInputError(
                "data_source must be auto, yahoo, or sec"
            )
        if self.data_source == "sec" and not (
            self.sec_user_agent or os.environ.get("SEC_USER_AGENT")
        ):
            raise ModelInputError(
                "SEC data requires sec_user_agent or the SEC_USER_AGENT environment "
                "variable, for example 'Your Name your.email@example.com'"
            )
        if self.fcff_method not in {"normalized", "latest", "bottom_up"}:
            raise ModelInputError(
                "fcff_method must be normalized, latest, or bottom_up"
            )
        if self.forecast_years < 1:
            raise ModelInputError("forecast_years must be at least 1")
        if self.normalization_years < 1:
            raise ModelInputError("normalization_years must be at least 1")
        if self.simulations < 100:
            raise ModelInputError("simulations must be at least 100")
        if self.request_retries < 0:
            raise ModelInputError("request_retries must be non-negative")
        if self.retry_delay_seconds < 0:
            raise ModelInputError("retry_delay_seconds must be non-negative")

        rate_fields = (
            "terminal_growth_rate",
            "equity_risk_premium",
            "risk_free_rate_fallback",
            "wacc_floor",
            "minimum_equity_spread",
            "default_tax_rate",
            "maximum_tax_rate",
        )
        for name in rate_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ModelInputError(f"{name} must be finite")

        optional_rates = (
            "explicit_growth_rate",
            "risk_free_rate_override",
            "wacc_override",
        )
        for name in optional_rates:
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ModelInputError(f"{name} must be finite when supplied")

        if self.explicit_growth_rate is not None and self.explicit_growth_rate <= -1:
            raise ModelInputError("explicit_growth_rate must exceed -100%")
        if self.terminal_growth_rate <= -1:
            raise ModelInputError("terminal_growth_rate must exceed -100%")
        if not 0 <= self.default_tax_rate <= self.maximum_tax_rate <= 1:
            raise ModelInputError(
                "tax rates must satisfy 0 <= default <= maximum <= 1"
            )
        if self.equity_risk_premium < 0:
            raise ModelInputError("equity_risk_premium must be non-negative")
        if self.risk_free_rate_override is not None and self.risk_free_rate_override < 0:
            raise ModelInputError("risk_free_rate_override must be non-negative")
        if self.wacc_override is not None and self.wacc_override <= 0:
            raise ModelInputError("wacc_override must be positive")
        if self.target_debt_weight is not None and not 0 <= self.target_debt_weight < 1:
            raise ModelInputError("target_debt_weight must be in [0, 1)")
        if self.fcff_override is not None and self.fcff_override <= 0:
            raise ModelInputError("fcff_override must be positive")
        if not math.isclose(
            self.auto_growth_revenue_weight + self.auto_growth_fcff_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ModelInputError("auto growth weights must sum to 1")
        if min(self.wacc_std, self.explicit_growth_std, self.terminal_growth_std) < 0:
            raise ModelInputError("simulation standard deviations must be non-negative")
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ModelInputError("minimum_valid_fraction must be in (0, 1]")

        for name in (
            "auto_growth_bounds",
            "growth_bounds",
            "terminal_growth_bounds",
            "wacc_bounds",
        ):
            lower, upper = getattr(self, name)
            if lower >= upper:
                raise ModelInputError(f"{name} lower bound must be below upper bound")


@dataclass(frozen=True, slots=True)
class FinancialStatements:
    income_statement: pd.DataFrame
    cash_flow: pd.DataFrame
    balance_sheet: pd.DataFrame
    source: str


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    current_price: float
    shares_outstanding: float
    market_cap: float


@dataclass(frozen=True, slots=True)
class FinancialMetrics:
    components: pd.DataFrame
    selected_fcff: float
    selected_fcff_method: str
    latest_cash_fcff: float | None
    latest_bottom_up_fcff: float | None
    normalized_fcff: float | None
    latest_period: str
    revenue_cagr: float | None
    cash_fcff_cagr: float | None
    total_debt: float
    cash_and_investments: float
    interest_expense: float
    average_tax_rate: float
    warnings: tuple[str, ...]

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash_and_investments


@dataclass(frozen=True, slots=True)
class GrowthAssumption:
    rate: float
    source: str


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    beta: float
    correlation: float | None
    r_squared: float | None
    annual_volatility: float | None
    annual_systematic_volatility: float | None
    annual_idiosyncratic_volatility: float | None
    observations: int
    source: str


@dataclass(frozen=True, slots=True)
class CapitalCosts:
    risk_free_rate: float
    risk_free_source: str
    equity_risk_premium: float
    adjusted_beta: float
    cost_of_equity: float
    pre_tax_cost_of_debt: float
    wacc_before_floor: float
    wacc: float
    wacc_source: str
    equity_weight: float
    debt_weight: float


@dataclass(frozen=True, slots=True)
class DCFValuation:
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    explicit_period_value: float
    terminal_value_present_value: float


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    mean: float
    median: float
    standard_deviation: float
    percentile_10: float
    percentile_90: float
    valid_scenarios: int
    rejected_scenarios: int


@dataclass(frozen=True, slots=True)
class ValuationResult:
    ticker: str
    data_source: str
    current_price: float
    point_estimate: DCFValuation
    simulation: SimulationSummary
    risk: RiskMetrics
    capital_costs: CapitalCosts
    financials: FinancialMetrics
    growth: GrowthAssumption
    terminal_growth_rate: float
    forecast_years: int
    reverse_dcf_growth: float | None
    sensitivity: tuple[tuple[str, tuple[float, ...]], ...]
    sensitivity_waccs: tuple[float, ...]
    warnings: tuple[str, ...]


class MarketDataProvider(Protocol):
    """Replaceable boundary for market and financial data."""

    def get_financial_statements(self, symbol: str) -> FinancialStatements: ...

    def get_price_history(
        self, symbol: str, *, period: str, interval: str
    ) -> pd.DataFrame: ...

    def get_market_snapshot(self, symbol: str) -> MarketSnapshot: ...


class YahooFinanceProvider:
    """Yahoo Finance adapter.

    Yahoo is convenient for a portfolio demonstration but is not treated as a
    guaranteed accounting source. Empty responses are retried and then rejected
    with an actionable error. The rest of the model does not depend on yfinance.
    """

    def __init__(self, *, retries: int = 2, retry_delay_seconds: float = 1.0):
        try:
            import yfinance as yf
        except ImportError as error:  # pragma: no cover - environment dependent
            raise DataUnavailableError(
                "yfinance is not installed; run: pip install yfinance pandas numpy"
            ) from error
        try:
            cache_dir = Path(tempfile.gettempdir()) / "monte_carlo_dcf_yfinance_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(cache_dir))
        except (AttributeError, OSError):
            # Older yfinance versions may not expose cache configuration.
            pass
        self._yf = yf
        self._retries = retries
        self._retry_delay_seconds = retry_delay_seconds
        self._tickers: dict[str, Any] = {}

    def _ticker(self, symbol: str) -> Any:
        normalized = symbol.strip().upper()
        if normalized not in self._tickers:
            self._tickers[normalized] = self._yf.Ticker(normalized)
        return self._tickers[normalized]

    def _with_retries(self, label: str, loader: Any) -> Any:
        errors: list[str] = []
        for attempt in range(self._retries + 1):
            try:
                value = loader()
                if value is not None:
                    if not isinstance(value, pd.DataFrame) or not value.empty:
                        return value
                errors.append("empty response")
            except Exception as error:  # pragma: no cover - external service
                errors.append(str(error))
            if attempt < self._retries:
                time.sleep(self._retry_delay_seconds * (attempt + 1))
        detail = errors[-1] if errors else "unknown error"
        raise DataUnavailableError(
            f"{label} unavailable after {self._retries + 1} attempt(s): {detail}. "
            "Yahoo may be rate-limiting the request; retry later or use cached/SEC data."
        )

    def _statement(self, symbol: str, attribute: str, getter: str) -> pd.DataFrame:
        ticker = self._ticker(symbol)

        def load() -> pd.DataFrame:
            direct = getattr(ticker, attribute, None)
            if isinstance(direct, pd.DataFrame) and not direct.empty:
                return direct
            method = getattr(ticker, getter, None)
            if callable(method):
                value = method(freq="yearly")
                if isinstance(value, pd.DataFrame):
                    return value
            return pd.DataFrame()

        return self._with_retries(f"{symbol} {attribute}", load)

    def get_financial_statements(self, symbol: str) -> FinancialStatements:
        return FinancialStatements(
            income_statement=self._statement(
                symbol, "income_stmt", "get_income_stmt"
            ),
            cash_flow=self._statement(symbol, "cashflow", "get_cash_flow"),
            balance_sheet=self._statement(
                symbol, "balance_sheet", "get_balance_sheet"
            ),
            source="Yahoo Finance",
        )

    def get_price_history(
        self, symbol: str, *, period: str, interval: str
    ) -> pd.DataFrame:
        history = self._with_retries(
            f"{symbol} price history",
            lambda: self._ticker(symbol).history(period=period, interval=interval),
        )
        if "Close" not in history.columns or history["Close"].dropna().empty:
            raise DataUnavailableError(f"closing prices are unavailable for {symbol}")
        return history

    def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        ticker = self._ticker(symbol)

        def fast_value(key: str) -> float | None:
            try:
                value = ticker.fast_info[key]
                number = float(value)
                return number if math.isfinite(number) else None
            except (KeyError, TypeError, ValueError):
                return None

        current_price = fast_value("last_price")
        market_cap = fast_value("market_cap")
        shares = fast_value("shares")

        if current_price is None or current_price <= 0:
            history = self.get_price_history(symbol, period="5d", interval="1d")
            current_price = float(history["Close"].dropna().iloc[-1])

        if shares is None or shares <= 0:
            try:
                info_shares = ticker.info.get("sharesOutstanding")
                shares = float(info_shares) if info_shares else None
            except Exception:  # pragma: no cover - external service
                shares = None

        if market_cap is None and shares is not None:
            market_cap = current_price * shares
        if shares is None and market_cap is not None and current_price > 0:
            shares = market_cap / current_price

        if shares is None or shares <= 0 or market_cap is None or market_cap <= 0:
            raise DataUnavailableError(
                f"shares outstanding or market capitalization unavailable for {symbol}"
            )
        return MarketSnapshot(current_price, shares, market_cap)


class SECCompanyFactsProvider:
    """Use official SEC XBRL facts for statements and Yahoo for market data.

    The SEC requires an identifying User-Agent. This adapter intentionally
    covers common US-GAAP concepts rather than silently guessing custom tags.
    Unsupported issuers can use the Yahoo fallback through data_source='auto'.
    """

    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    CONCEPTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "EBIT": ("OperatingIncomeLoss",),
        "Tax Provision": ("IncomeTaxExpenseBenefit",),
        "Pretax Income": (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ),
        "Interest Expense": (
            "InterestExpenseNonOperating",
            "InterestAndDebtExpense",
            "InterestExpenseDebt",
            "InterestExpense",
        ),
        "Total Revenue": (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
        ),
        "Operating Cash Flow": ("NetCashProvidedByUsedInOperatingActivities",),
        "Capital Expenditure": (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsForAdditionsToPropertyPlantAndEquipment",
        ),
        "Depreciation And Amortization": (
            "DepreciationDepletionAndAmortization",
            "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
            "Depreciation",
        ),
        "Cash And Cash Equivalents": (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        "Short Term Investments": (
            "ShortTermInvestments",
            "MarketableSecuritiesCurrent",
        ),
        "Current Debt": (
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "LongTermDebtAndCapitalLeaseObligationsCurrent",
            "LongTermDebtCurrent",
            "ShortTermBorrowings",
        ),
        "Long Term Debt": (
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "LongTermDebtAndCapitalLeaseObligations",
            "LongTermDebtNoncurrent",
            "LongTermDebt",
        ),
    }

    INSTANT_ROWS = {
        "Cash And Cash Equivalents",
        "Short Term Investments",
        "Current Debt",
        "Long Term Debt",
    }

    def __init__(
        self,
        market_provider: MarketDataProvider,
        *,
        user_agent: str,
        retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if not user_agent.strip():
            raise ModelInputError("SEC user agent must not be empty")
        self.market_provider = market_provider
        self.user_agent = user_agent.strip()
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._ticker_to_cik: dict[str, str] | None = None

    def _get_json(self, url: str) -> dict[str, Any]:
        last_error = "unknown error"
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict):
                    return payload
                last_error = "response was not a JSON object"
            except (OSError, ValueError, urllib.error.URLError) as error:
                last_error = str(error)
            if attempt < self.retries:
                time.sleep(self.retry_delay_seconds * (attempt + 1))
        raise DataUnavailableError(f"SEC request failed for {url}: {last_error}")

    def _cik_for_ticker(self, ticker: str) -> str:
        if self._ticker_to_cik is None:
            payload = self._get_json(self.TICKER_MAP_URL)
            mapping: dict[str, str] = {}
            for item in payload.values():
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("ticker", "")).upper()
                cik = item.get("cik_str")
                if symbol and cik is not None:
                    mapping[symbol] = f"{int(cik):010d}"
            self._ticker_to_cik = mapping
        try:
            return self._ticker_to_cik[ticker.upper()]
        except KeyError as error:
            raise DataUnavailableError(
                f"{ticker} was not found in the SEC ticker-to-CIK mapping"
            ) from error

    @staticmethod
    def _annual_series(
        facts: dict[str, Any], concepts: Sequence[str], *, instant: bool
    ) -> pd.Series | None:
        candidates: list[pd.Series] = []
        for concept in concepts:
            concept_data = facts.get(concept)
            if not isinstance(concept_data, dict):
                continue
            unit_groups = concept_data.get("units", {})
            observations = unit_groups.get("USD")
            if not isinstance(observations, list):
                continue
            selected: dict[pd.Timestamp, tuple[pd.Timestamp, float]] = {}
            for observation in observations:
                if observation.get("form") not in {"10-K", "10-K/A", "20-F", "40-F"}:
                    continue
                if not instant and not observation.get("start"):
                    continue
                if not instant and observation.get("fp") not in {"FY", None}:
                    continue
                try:
                    end = pd.Timestamp(observation["end"])
                    if not instant:
                        start = pd.Timestamp(observation["start"])
                        duration_days = (end - start).days
                        if not 250 <= duration_days <= 450:
                            continue
                    filed = pd.Timestamp(observation.get("filed", observation["end"]))
                    value = float(observation["val"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                current = selected.get(end)
                if current is None or filed >= current[0]:
                    selected[end] = (filed, value)
            if selected:
                candidates.append(
                    pd.Series(
                        {end: item[1] for end, item in sorted(selected.items())},
                        dtype=float,
                    )
                )
        if not candidates:
            return None
        return max(candidates, key=lambda series: (series.index.max(), len(series)))

    @classmethod
    def statements_from_companyfacts(
        cls, payload: dict[str, Any]
    ) -> FinancialStatements:
        namespace = payload.get("facts", {}).get("us-gaap", {})
        if not isinstance(namespace, dict) or not namespace:
            raise DataUnavailableError(
                "SEC filing has no supported us-gaap company facts"
            )
        rows: dict[str, pd.Series] = {}
        for label, concepts in cls.CONCEPTS.items():
            series = cls._annual_series(
                namespace, concepts, instant=label in cls.INSTANT_ROWS
            )
            if series is not None:
                rows[label] = series

        long_term_total = cls._annual_series(
            namespace,
            (
                "LongTermDebtAndFinanceLeaseObligationsIncludingCurrentMaturities",
                "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
            ),
            instant=True,
        )
        short_term_borrowings = cls._annual_series(
            namespace, ("ShortTermBorrowings",), instant=True
        )
        if long_term_total is not None:
            total_debt = long_term_total
            if short_term_borrowings is not None:
                total_debt = total_debt.add(short_term_borrowings, fill_value=0.0)
            rows["Total Debt"] = total_debt

        def frame(names: Sequence[str]) -> pd.DataFrame:
            available = {name: rows[name] for name in names if name in rows}
            if not available:
                return pd.DataFrame()
            return pd.DataFrame(available).T.sort_index(axis=1)

        income = frame(
            (
                "EBIT",
                "Tax Provision",
                "Pretax Income",
                "Interest Expense",
                "Total Revenue",
            )
        )
        cash_flow = frame(
            (
                "Operating Cash Flow",
                "Capital Expenditure",
                "Depreciation And Amortization",
            )
        )
        balance = frame(
            (
                "Cash And Cash Equivalents",
                "Short Term Investments",
                "Total Debt",
                "Current Debt",
                "Long Term Debt",
            )
        )
        for name, value in (
            ("income statement", income),
            ("cash-flow statement", cash_flow),
            ("balance sheet", balance),
        ):
            if value.empty:
                raise DataUnavailableError(f"SEC {name} could not be constructed")
        return FinancialStatements(
            income_statement=income,
            cash_flow=cash_flow,
            balance_sheet=balance,
            source="SEC EDGAR Company Facts (financials) + Yahoo Finance (market data)",
        )

    def get_financial_statements(self, symbol: str) -> FinancialStatements:
        cik = self._cik_for_ticker(symbol)
        payload = self._get_json(self.COMPANY_FACTS_URL.format(cik=cik))
        return self.statements_from_companyfacts(payload)

    def get_price_history(
        self, symbol: str, *, period: str, interval: str
    ) -> pd.DataFrame:
        return self.market_provider.get_price_history(
            symbol, period=period, interval=interval
        )

    def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return self.market_provider.get_market_snapshot(symbol)


class FallbackFinancialProvider:
    """Try official SEC statements, then fall back to Yahoo in auto mode."""

    def __init__(
        self,
        primary: MarketDataProvider,
        fallback: MarketDataProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def get_financial_statements(self, symbol: str) -> FinancialStatements:
        try:
            return self.primary.get_financial_statements(symbol)
        except ValuationError as error:
            statements = self.fallback.get_financial_statements(symbol)
            return FinancialStatements(
                statements.income_statement,
                statements.cash_flow,
                statements.balance_sheet,
                f"{statements.source}; SEC fallback reason: {error}",
            )

    def get_price_history(
        self, symbol: str, *, period: str, interval: str
    ) -> pd.DataFrame:
        return self.fallback.get_price_history(
            symbol, period=period, interval=interval
        )

    def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return self.fallback.get_market_snapshot(symbol)


class FinancialStatementProcessor:
    """Validate statements and calculate two independent FCFF series."""

    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "ebit": ("EBIT", "Operating Income"),
        "tax_provision": ("Tax Provision", "Income Tax Expense"),
        "pretax_income": ("Pretax Income", "Income Before Tax"),
        "depreciation": (
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
            "Depreciation",
        ),
        "capital_expenditure": (
            "Capital Expenditure",
            "Capital Expenditures",
            "Purchase Of PPE",
        ),
        "change_working_capital": (
            "Change In Working Capital",
            "Changes In Working Capital",
        ),
        "operating_cash_flow": (
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
            "Cash Flow From Continuing Operating Activities",
        ),
        "revenue": ("Total Revenue", "Operating Revenue", "Revenue"),
        "interest_expense": (
            "Interest Expense",
            "Interest Expense Non Operating",
            "Interest And Debt Expense",
        ),
        "total_debt": (
            "Total Debt",
            "Long Term Debt And Capital Lease Obligation",
        ),
        "current_debt": (
            "Current Debt",
            "Short Term Debt",
            "Short Term Borrowings",
        ),
        "long_term_debt": (
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
            "Non Current Debt",
        ),
        "cash_and_short_term_investments": (
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
        ),
        "cash": (
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents",
            "Cash Financial",
        ),
        "short_term_investments": (
            "Other Short Term Investments",
            "Short Term Investments",
            "Current Marketable Securities",
        ),
    }

    def __init__(self, config: ValuationConfig):
        self.config = config

    @staticmethod
    def _clean(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            raise DataUnavailableError(f"{name} is empty")
        cleaned = frame.copy()
        converted = pd.to_datetime(cleaned.columns, errors="coerce")
        if not pd.isna(converted).any():
            cleaned.columns = converted
        cleaned = cleaned.sort_index(axis=1)
        cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
        if cleaned.dropna(how="all").empty:
            raise DataUnavailableError(f"{name} contains no numeric observations")
        return cleaned

    @classmethod
    def _row(cls, frame: pd.DataFrame, metric: str) -> pd.Series | None:
        for alias in cls.ALIASES[metric]:
            if alias in frame.index:
                row = frame.loc[alias]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return pd.to_numeric(row, errors="coerce").sort_index()
        return None

    @classmethod
    def _required_row(
        cls, frame: pd.DataFrame, metric: str, statement_name: str
    ) -> pd.Series:
        row = cls._row(frame, metric)
        if row is None or row.dropna().empty:
            aliases = ", ".join(cls.ALIASES[metric])
            raise DataUnavailableError(
                f"{metric} missing from {statement_name}; tried: {aliases}"
            )
        return row

    @classmethod
    def _latest_value(
        cls, frame: pd.DataFrame, metric: str, default: float | None = None
    ) -> float:
        row = cls._row(frame, metric)
        if row is None or row.dropna().empty:
            if default is None:
                raise DataUnavailableError(f"{metric} is unavailable")
            return default
        return float(row.dropna().iloc[-1])

    @staticmethod
    def _cagr(series: pd.Series | None, observations: int = 5) -> float | None:
        if series is None:
            return None
        values = pd.to_numeric(series, errors="coerce").dropna().tail(observations)
        if len(values) < 2 or values.iloc[0] <= 0 or values.iloc[-1] <= 0:
            return None
        periods = len(values) - 1
        value = float((values.iloc[-1] / values.iloc[0]) ** (1 / periods) - 1)
        return value if math.isfinite(value) else None

    @staticmethod
    def _relative_gap(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        scale = max(abs(left), abs(right), 1e-12)
        return abs(left - right) / scale

    def process(self, statements: FinancialStatements) -> FinancialMetrics:
        income = self._clean(statements.income_statement, "income statement")
        cash_flow = self._clean(statements.cash_flow, "cash-flow statement")
        balance = self._clean(statements.balance_sheet, "balance sheet")
        warnings: list[str] = []

        tax_provision = self._required_row(income, "tax_provision", "income statement")
        pretax_income = self._required_row(income, "pretax_income", "income statement")
        tax_frame = pd.concat(
            [tax_provision.rename("tax"), pretax_income.rename("pretax")], axis=1
        ).dropna(how="all")
        raw_tax = tax_frame["tax"].abs() / tax_frame["pretax"].abs().where(
            tax_frame["pretax"].abs() > 1e-12
        )
        invalid_tax = raw_tax.isna() | (raw_tax < 0) | (
            raw_tax > self.config.maximum_tax_rate
        )
        if invalid_tax.any():
            dates = ", ".join(str(value)[:10] for value in raw_tax.index[invalid_tax])
            warnings.append(
                "Invalid effective tax observations were replaced by the default "
                f"rate for: {dates}."
            )
        tax_rate = raw_tax.where(~invalid_tax, self.config.default_tax_rate)
        recent_tax = tax_rate.dropna().tail(3)
        average_tax_rate = (
            float(recent_tax.mean())
            if not recent_tax.empty
            else self.config.default_tax_rate
        )

        operating_cf = self._required_row(
            cash_flow, "operating_cash_flow", "cash-flow statement"
        )
        capex = self._required_row(
            cash_flow, "capital_expenditure", "cash-flow statement"
        )
        interest = self._row(income, "interest_expense")
        if interest is None:
            interest = pd.Series(0.0, index=operating_cf.index)
            warnings.append(
                "Interest expense was unavailable; cash-flow FCFF equals CFO less CapEx."
            )

        cash_components = pd.concat(
            [
                operating_cf.rename("operating_cash_flow"),
                capex.rename("capital_expenditure"),
                interest.abs().rename("interest_expense"),
                tax_rate.rename("tax_rate"),
            ],
            axis=1,
        ).sort_index()
        cash_components["tax_rate"] = cash_components["tax_rate"].fillna(
            average_tax_rate
        )
        cash_components = cash_components.dropna(
            subset=["operating_cash_flow", "capital_expenditure"]
        )
        cash_components["interest_expense"] = cash_components[
            "interest_expense"
        ].fillna(0.0)
        cash_components["cash_fcff"] = (
            cash_components["operating_cash_flow"]
            - cash_components["capital_expenditure"].abs()
            + cash_components["interest_expense"]
            * (1 - cash_components["tax_rate"])
        )
        cash_fcff = cash_components["cash_fcff"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        ebit = self._row(income, "ebit")
        depreciation = self._row(cash_flow, "depreciation")
        working_capital = self._row(cash_flow, "change_working_capital")
        bottom_up_fcff = pd.Series(dtype=float)
        bottom_components = pd.DataFrame()
        if ebit is None:
            warnings.append("EBIT was unavailable; bottom-up FCFF was not calculated.")
        elif depreciation is None or working_capital is None:
            missing = []
            if depreciation is None:
                missing.append("depreciation")
            if working_capital is None:
                missing.append("change in working capital")
            warnings.append(
                "Bottom-up FCFF was not calculated because "
                + " and ".join(missing)
                + " were unavailable; cash-flow FCFF remains the primary measure."
            )
        else:
            bottom_components = pd.concat(
                [
                    ebit.rename("ebit"),
                    depreciation.rename("depreciation"),
                    capex.rename("capital_expenditure_bottom_up"),
                    working_capital.rename("change_working_capital"),
                    tax_rate.rename("tax_rate_bottom_up"),
                ],
                axis=1,
            ).sort_index()
            bottom_components["tax_rate_bottom_up"] = bottom_components[
                "tax_rate_bottom_up"
            ].fillna(average_tax_rate)
            bottom_components[["depreciation", "change_working_capital"]] = (
                bottom_components[["depreciation", "change_working_capital"]].fillna(
                    0.0
                )
            )
            bottom_components = bottom_components.dropna(
                subset=["ebit", "capital_expenditure_bottom_up"]
            )
            bottom_components["bottom_up_fcff"] = (
                bottom_components["ebit"]
                * (1 - bottom_components["tax_rate_bottom_up"])
                + bottom_components["depreciation"]
                - bottom_components["capital_expenditure_bottom_up"].abs()
                + bottom_components["change_working_capital"]
            )
            bottom_up_fcff = bottom_components["bottom_up_fcff"].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

        components = cash_components.join(bottom_components, how="outer")
        positive_cash = cash_fcff[cash_fcff > 0].tail(
            self.config.normalization_years
        )
        latest_cash_fcff = float(cash_fcff.iloc[-1]) if not cash_fcff.empty else None
        normalized_fcff = (
            float(positive_cash.median()) if not positive_cash.empty else None
        )
        latest_bottom_up_fcff = (
            float(bottom_up_fcff.iloc[-1]) if not bottom_up_fcff.empty else None
        )

        gap = self._relative_gap(latest_cash_fcff, latest_bottom_up_fcff)
        if gap is not None and gap > self.config.fcff_divergence_warning:
            warnings.append(
                "Cash-flow and bottom-up FCFF differ by "
                f"{gap:.1%}; inspect non-cash and one-off items before relying on the model."
            )
        latest_gap = self._relative_gap(latest_cash_fcff, normalized_fcff)
        if latest_gap is not None and latest_gap > self.config.latest_to_normalized_warning:
            warnings.append(
                "Latest cash-flow FCFF differs materially from its recent median; "
                "the normalized anchor was retained."
            )

        if self.config.fcff_override is not None:
            selected_fcff = self.config.fcff_override
            selected_method = "manual analyst override"
        elif self.config.fcff_method == "normalized":
            if normalized_fcff is None:
                raise DataUnavailableError(
                    "no positive cash-flow FCFF observations are available for normalization"
                )
            selected_fcff = normalized_fcff
            selected_method = (
                f"{len(positive_cash)}-period median cash-flow FCFF"
            )
        elif self.config.fcff_method == "latest":
            if latest_cash_fcff is None or latest_cash_fcff <= 0:
                raise ModelInputError(
                    "latest cash-flow FCFF is non-positive; use normalized or an override"
                )
            selected_fcff = latest_cash_fcff
            selected_method = "latest cash-flow FCFF"
        else:
            if latest_bottom_up_fcff is None or latest_bottom_up_fcff <= 0:
                raise ModelInputError(
                    "latest bottom-up FCFF is non-positive; use normalized or an override"
                )
            selected_fcff = latest_bottom_up_fcff
            selected_method = "latest bottom-up FCFF"

        total_debt = abs(self._latest_value(balance, "total_debt", default=np.nan))
        if np.isnan(total_debt):
            current_debt = abs(self._latest_value(balance, "current_debt", 0.0))
            long_term_debt = abs(
                self._latest_value(balance, "long_term_debt", 0.0)
            )
            total_debt = current_debt + long_term_debt
            warnings.append("Total debt was reconstructed from current and long-term debt.")

        combined_cash = self._latest_value(
            balance, "cash_and_short_term_investments", default=np.nan
        )
        if np.isnan(combined_cash):
            cash = max(0.0, self._latest_value(balance, "cash", 0.0))
            investments = max(
                0.0, self._latest_value(balance, "short_term_investments", 0.0)
            )
            cash_and_investments = cash + investments
        else:
            cash_and_investments = max(0.0, combined_cash)

        latest_interest = abs(self._latest_value(income, "interest_expense", 0.0))
        revenue = self._row(income, "revenue")
        revenue_cagr = self._cagr(revenue)
        cash_fcff_cagr = self._cagr(cash_fcff)
        if revenue_cagr is None:
            warnings.append(
                "Revenue CAGR could not be calculated; auto growth will use another anchor."
            )
        if cash_fcff_cagr is None:
            warnings.append(
                "Cash-flow FCFF CAGR could not be calculated from positive endpoints."
            )

        latest_index = components.dropna(how="all").index[-1]
        latest_period = str(latest_index)[:10]
        return FinancialMetrics(
            components=components,
            selected_fcff=float(selected_fcff),
            selected_fcff_method=selected_method,
            latest_cash_fcff=latest_cash_fcff,
            latest_bottom_up_fcff=latest_bottom_up_fcff,
            normalized_fcff=normalized_fcff,
            latest_period=latest_period,
            revenue_cagr=revenue_cagr,
            cash_fcff_cagr=cash_fcff_cagr,
            total_debt=float(total_debt),
            cash_and_investments=float(cash_and_investments),
            interest_expense=float(latest_interest),
            average_tax_rate=average_tax_rate,
            warnings=tuple(warnings),
        )


class MarketRiskAnalyzer:
    PERIODS_PER_YEAR: ClassVar[dict[str, int]] = {
        "1d": 252,
        "1wk": 52,
        "1w": 52,
        "1mo": 12,
    }

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def analyze(
        self, symbol: str, benchmark: str, *, period: str, interval: str
    ) -> RiskMetrics:
        if interval not in self.PERIODS_PER_YEAR:
            raise ModelInputError(f"unsupported risk interval: {interval}")
        stock = self.provider.get_price_history(symbol, period=period, interval=interval)
        market = self.provider.get_price_history(
            benchmark, period=period, interval=interval
        )
        prices = pd.concat(
            [stock["Close"].rename("stock"), market["Close"].rename("market")],
            axis=1,
            join="inner",
        ).dropna()
        returns = prices.pct_change(fill_method=None).dropna()
        if len(returns) < 24:
            raise DataUnavailableError(
                f"only {len(returns)} aligned returns; at least 24 are required"
            )
        market_variance = float(returns["market"].var(ddof=1))
        if market_variance <= 0:
            raise ModelInputError("benchmark return variance must be positive")
        beta = float(returns["stock"].cov(returns["market"]) / market_variance)
        correlation = float(returns["stock"].corr(returns["market"]))
        alpha = float(returns["stock"].mean() - beta * returns["market"].mean())
        residuals = returns["stock"] - alpha - beta * returns["market"]
        periods = self.PERIODS_PER_YEAR[interval]
        stock_volatility = float(returns["stock"].std(ddof=1) * np.sqrt(periods))
        market_volatility = float(returns["market"].std(ddof=1) * np.sqrt(periods))
        return RiskMetrics(
            beta=beta,
            correlation=correlation,
            r_squared=float(np.clip(correlation**2, 0.0, 1.0)),
            annual_volatility=stock_volatility,
            annual_systematic_volatility=abs(beta) * market_volatility,
            annual_idiosyncratic_volatility=float(
                residuals.std(ddof=1) * np.sqrt(periods)
            ),
            observations=len(returns),
            source=f"{period} regression versus {benchmark}",
        )


class CapitalCostModel:
    def __init__(
        self, provider: MarketDataProvider, config: ValuationConfig
    ) -> None:
        self.provider = provider
        self.config = config

    def _risk_free_rate(self) -> tuple[float, str]:
        if self.config.risk_free_rate_override is not None:
            return self.config.risk_free_rate_override, "manual override"
        try:
            history = self.provider.get_price_history(
                self.config.risk_free_symbol, period="1mo", interval="1d"
            )
            observations = history["Close"].dropna().tail(5)
            if observations.empty:
                raise DataUnavailableError("risk-free history is empty")
            return float(observations.mean() / 100), self.config.risk_free_symbol
        except ValuationError:
            return self.config.risk_free_rate_fallback, "configured fallback"

    @staticmethod
    def _adjust_beta(
        raw_beta: float,
        debt: float,
        equity: float,
        tax_rate: float,
        target_debt_weight: float | None,
    ) -> float:
        if equity <= 0:
            raise ModelInputError("market value of equity must be positive")
        if target_debt_weight is None:
            return raw_beta
        unlevered = raw_beta / (1 + (1 - tax_rate) * debt / equity)
        target_debt_to_equity = target_debt_weight / (1 - target_debt_weight)
        return unlevered * (1 + (1 - tax_rate) * target_debt_to_equity)

    def calculate(
        self,
        risk: RiskMetrics,
        financials: FinancialMetrics,
        snapshot: MarketSnapshot,
    ) -> CapitalCosts:
        risk_free_rate, risk_free_source = self._risk_free_rate()
        adjusted_beta = self._adjust_beta(
            risk.beta,
            financials.total_debt,
            snapshot.market_cap,
            financials.average_tax_rate,
            self.config.target_debt_weight,
        )
        capm_cost = risk_free_rate + adjusted_beta * self.config.equity_risk_premium
        cost_of_equity = max(
            capm_cost, risk_free_rate + self.config.minimum_equity_spread
        )
        if financials.total_debt > 0:
            raw_debt_cost = financials.interest_expense / financials.total_debt
            debt_cost = max(raw_debt_cost, risk_free_rate)
        else:
            debt_cost = 0.0

        total_capital = snapshot.market_cap + financials.total_debt
        if total_capital <= 0:
            raise ModelInputError("total capital must be positive")
        equity_weight = snapshot.market_cap / total_capital
        debt_weight = financials.total_debt / total_capital
        raw_wacc = (
            equity_weight * cost_of_equity
            + debt_weight
            * debt_cost
            * (1 - financials.average_tax_rate)
        )
        if self.config.wacc_override is not None:
            wacc = self.config.wacc_override
            wacc_source = "manual override"
        else:
            wacc = max(raw_wacc, self.config.wacc_floor)
            wacc_source = (
                "calculated CAPM/WACC"
                if wacc == raw_wacc
                else f"{self.config.wacc_floor:.2%} configured floor"
            )
        return CapitalCosts(
            risk_free_rate=risk_free_rate,
            risk_free_source=risk_free_source,
            equity_risk_premium=self.config.equity_risk_premium,
            adjusted_beta=adjusted_beta,
            cost_of_equity=cost_of_equity,
            pre_tax_cost_of_debt=debt_cost,
            wacc_before_floor=raw_wacc,
            wacc=wacc,
            wacc_source=wacc_source,
            equity_weight=equity_weight,
            debt_weight=debt_weight,
        )


class DCFModel:
    @staticmethod
    def value(
        *,
        current_fcff: float,
        wacc: float,
        explicit_growth_rate: float,
        terminal_growth_rate: float,
        forecast_years: int,
        shares_outstanding: float,
        net_debt: float,
    ) -> DCFValuation:
        if current_fcff <= 0:
            raise ModelInputError("current FCFF must be positive")
        if shares_outstanding <= 0:
            raise ModelInputError("shares outstanding must be positive")
        if wacc <= terminal_growth_rate:
            raise ModelInputError("WACC must exceed terminal growth")
        if min(1 + wacc, 1 + explicit_growth_rate, 1 + terminal_growth_rate) <= 0:
            raise ModelInputError("rates must exceed -100%")

        years = np.arange(1, forecast_years + 1, dtype=float)
        future_fcff = current_fcff * (1 + explicit_growth_rate) ** years
        explicit_value = float((future_fcff / (1 + wacc) ** years).sum())
        terminal_value = (
            future_fcff[-1]
            * (1 + terminal_growth_rate)
            / (wacc - terminal_growth_rate)
        )
        terminal_present_value = float(
            terminal_value / (1 + wacc) ** forecast_years
        )
        enterprise_value = explicit_value + terminal_present_value
        equity_value = enterprise_value - net_debt
        intrinsic = equity_value / shares_outstanding
        if not math.isfinite(intrinsic):
            raise ModelInputError("DCF produced a non-finite value")
        return DCFValuation(
            intrinsic_value_per_share=float(intrinsic),
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            explicit_period_value=explicit_value,
            terminal_value_present_value=terminal_present_value,
        )


class MonteCarloSimulator:
    def __init__(self, config: ValuationConfig):
        self.config = config

    def run(
        self,
        *,
        current_fcff: float,
        base_wacc: float,
        base_growth: float,
        shares_outstanding: float,
        net_debt: float,
    ) -> SimulationSummary:
        config = self.config
        rng = np.random.default_rng(config.random_seed)
        wacc = rng.normal(base_wacc, config.wacc_std, config.simulations)
        growth = rng.normal(
            base_growth, config.explicit_growth_std, config.simulations
        )
        terminal_growth = rng.normal(
            config.terminal_growth_rate,
            config.terminal_growth_std,
            config.simulations,
        )
        mask = (
            (wacc >= config.wacc_bounds[0])
            & (wacc <= config.wacc_bounds[1])
            & (growth >= config.growth_bounds[0])
            & (growth <= config.growth_bounds[1])
            & (terminal_growth >= config.terminal_growth_bounds[0])
            & (terminal_growth <= config.terminal_growth_bounds[1])
            & (wacc - terminal_growth >= config.min_terminal_spread)
        )
        wacc = wacc[mask]
        growth = growth[mask]
        terminal_growth = terminal_growth[mask]
        if len(wacc) == 0:
            raise ModelInputError("all Monte Carlo scenarios violated constraints")

        years = np.arange(1, config.forecast_years + 1, dtype=float)
        fcff = current_fcff * (1 + growth[:, None]) ** years
        explicit = (fcff / (1 + wacc[:, None]) ** years).sum(axis=1)
        terminal = fcff[:, -1] * (1 + terminal_growth) / (
            wacc - terminal_growth
        )
        enterprise = explicit + terminal / (1 + wacc) ** config.forecast_years
        prices = (enterprise - net_debt) / shares_outstanding
        prices = prices[np.isfinite(prices) & (prices > 0)]
        rejected = config.simulations - len(prices)
        if len(prices) / config.simulations < config.minimum_valid_fraction:
            raise ModelInputError(
                "too many Monte Carlo scenarios were rejected; review assumptions"
            )
        return SimulationSummary(
            mean=float(np.mean(prices)),
            median=float(np.median(prices)),
            standard_deviation=float(np.std(prices, ddof=1)),
            percentile_10=float(np.percentile(prices, 10)),
            percentile_90=float(np.percentile(prices, 90)),
            valid_scenarios=len(prices),
            rejected_scenarios=rejected,
        )


def select_growth(
    config: ValuationConfig, financials: FinancialMetrics
) -> GrowthAssumption:
    if config.explicit_growth_rate is not None:
        return GrowthAssumption(config.explicit_growth_rate, "manual analyst override")

    revenue = financials.revenue_cagr
    fcff = financials.cash_fcff_cagr
    if revenue is not None and fcff is not None:
        raw = (
            config.auto_growth_revenue_weight * revenue
            + config.auto_growth_fcff_weight * fcff
        )
        source = (
            f"auto: {config.auto_growth_revenue_weight:.0%} revenue CAGR + "
            f"{config.auto_growth_fcff_weight:.0%} cash-FCFF CAGR"
        )
    elif revenue is not None:
        raw = revenue
        source = "auto: revenue CAGR (FCFF CAGR unavailable)"
    elif fcff is not None:
        raw = fcff
        source = "auto: cash-FCFF CAGR (revenue CAGR unavailable)"
    else:
        raw = config.auto_growth_fallback
        source = "configured fallback; historical CAGRs unavailable"

    lower, upper = config.auto_growth_bounds
    selected = float(np.clip(raw, lower, upper))
    if selected != raw:
        source += f", capped to [{lower:.1%}, {upper:.1%}]"
    return GrowthAssumption(selected, source)


def reverse_dcf_growth(
    *,
    target_price: float,
    current_fcff: float,
    wacc: float,
    terminal_growth_rate: float,
    forecast_years: int,
    shares_outstanding: float,
    net_debt: float,
    lower: float = -0.50,
    upper: float = 1.00,
) -> float | None:
    def difference(growth: float) -> float:
        value = DCFModel.value(
            current_fcff=current_fcff,
            wacc=wacc,
            explicit_growth_rate=growth,
            terminal_growth_rate=terminal_growth_rate,
            forecast_years=forecast_years,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
        ).intrinsic_value_per_share
        return value - target_price

    low_value = difference(lower)
    high_value = difference(upper)
    if low_value == 0:
        return lower
    if high_value == 0:
        return upper
    if low_value * high_value > 0:
        return None
    for _ in range(100):
        midpoint = (lower + upper) / 2
        mid_value = difference(midpoint)
        if abs(mid_value) < 1e-8:
            return midpoint
        if low_value * mid_value <= 0:
            upper = midpoint
            high_value = mid_value
        else:
            lower = midpoint
            low_value = mid_value
    return (lower + upper) / 2


class MonteCarloDCFEngine:
    def __init__(
        self,
        config: ValuationConfig,
        provider: MarketDataProvider | None = None,
    ) -> None:
        self.config = config
        if provider is not None:
            self.provider = provider
            return

        yahoo = YahooFinanceProvider(
            retries=config.request_retries,
            retry_delay_seconds=config.retry_delay_seconds,
        )
        sec_user_agent = config.sec_user_agent or os.environ.get("SEC_USER_AGENT")
        if config.data_source == "yahoo" or not sec_user_agent:
            self.provider = yahoo
            return

        sec = SECCompanyFactsProvider(
            yahoo,
            user_agent=sec_user_agent,
            retries=config.request_retries,
            retry_delay_seconds=config.retry_delay_seconds,
        )
        self.provider = (
            sec
            if config.data_source == "sec"
            else FallbackFinancialProvider(sec, yahoo)
        )

    def _risk(self, warnings: list[str]) -> RiskMetrics:
        if self.config.beta_override is not None:
            return RiskMetrics(
                beta=self.config.beta_override,
                correlation=None,
                r_squared=None,
                annual_volatility=None,
                annual_systematic_volatility=None,
                annual_idiosyncratic_volatility=None,
                observations=0,
                source="manual beta override",
            )
        try:
            return MarketRiskAnalyzer(self.provider).analyze(
                self.config.ticker,
                self.config.market_symbol,
                period=self.config.risk_period,
                interval=self.config.risk_interval,
            )
        except ValuationError as error:
            warnings.append(
                f"Beta regression failed ({error}); beta fallback "
                f"{self.config.beta_fallback:.2f} was used."
            )
            return RiskMetrics(
                beta=self.config.beta_fallback,
                correlation=None,
                r_squared=None,
                annual_volatility=None,
                annual_systematic_volatility=None,
                annual_idiosyncratic_volatility=None,
                observations=0,
                source="configured fallback",
            )

    def run(self) -> ValuationResult:
        config = self.config
        LOGGER.info("Fetching %s data", config.ticker)
        statements = self.provider.get_financial_statements(config.ticker)
        snapshot = self.provider.get_market_snapshot(config.ticker)
        financials = FinancialStatementProcessor(config).process(statements)
        warnings = list(financials.warnings)
        risk = self._risk(warnings)
        capital_costs = CapitalCostModel(self.provider, config).calculate(
            risk, financials, snapshot
        )
        if capital_costs.risk_free_source == "configured fallback":
            warnings.append(
                "Risk-free-rate download failed; the configured fallback was used."
            )
        if (
            config.wacc_override is None
            and capital_costs.wacc_before_floor < config.wacc_floor
        ):
            warnings.append(
                f"Calculated WACC was raised to the {config.wacc_floor:.2%} floor."
            )

        growth = select_growth(config, financials)
        if (
            financials.revenue_cagr is not None
            and growth.rate > financials.revenue_cagr + 0.02
        ):
            warnings.append(
                "Selected growth exceeds historical revenue CAGR by more than 2 percentage points."
            )

        point = DCFModel.value(
            current_fcff=financials.selected_fcff,
            wacc=capital_costs.wacc,
            explicit_growth_rate=growth.rate,
            terminal_growth_rate=config.terminal_growth_rate,
            forecast_years=config.forecast_years,
            shares_outstanding=snapshot.shares_outstanding,
            net_debt=financials.net_debt,
        )
        simulation = MonteCarloSimulator(config).run(
            current_fcff=financials.selected_fcff,
            base_wacc=capital_costs.wacc,
            base_growth=growth.rate,
            shares_outstanding=snapshot.shares_outstanding,
            net_debt=financials.net_debt,
        )
        implied_growth = reverse_dcf_growth(
            target_price=snapshot.current_price,
            current_fcff=financials.selected_fcff,
            wacc=capital_costs.wacc,
            terminal_growth_rate=config.terminal_growth_rate,
            forecast_years=config.forecast_years,
            shares_outstanding=snapshot.shares_outstanding,
            net_debt=financials.net_debt,
        )

        waccs = tuple(
            max(config.terminal_growth_rate + 0.005, capital_costs.wacc + delta)
            for delta in (-0.01, 0.0, 0.01)
        )
        sensitivity_rows: list[tuple[str, tuple[float, ...]]] = []
        for growth_delta in (-0.02, 0.0, 0.02):
            row_growth = max(-0.95, growth.rate + growth_delta)
            values = tuple(
                DCFModel.value(
                    current_fcff=financials.selected_fcff,
                    wacc=wacc,
                    explicit_growth_rate=row_growth,
                    terminal_growth_rate=config.terminal_growth_rate,
                    forecast_years=config.forecast_years,
                    shares_outstanding=snapshot.shares_outstanding,
                    net_debt=financials.net_debt,
                ).intrinsic_value_per_share
                for wacc in waccs
            )
            sensitivity_rows.append((f"{row_growth:.1%}", values))

        return ValuationResult(
            ticker=config.ticker,
            data_source=statements.source,
            current_price=snapshot.current_price,
            point_estimate=point,
            simulation=simulation,
            risk=risk,
            capital_costs=capital_costs,
            financials=financials,
            growth=growth,
            terminal_growth_rate=config.terminal_growth_rate,
            forecast_years=config.forecast_years,
            reverse_dcf_growth=implied_growth,
            sensitivity=tuple(sensitivity_rows),
            sensitivity_waccs=waccs,
            warnings=tuple(warnings),
        )


def _money_or_na(value: float | None, *, billions: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    if billions:
        return f"${value / 1e9:,.2f}B"
    return f"${value:,.2f}"


def _percent_or_na(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def format_report(result: ValuationResult) -> str:
    sim = result.simulation
    costs = result.capital_costs
    financials = result.financials
    risk = result.risk
    terminal_share = (
        result.point_estimate.terminal_value_present_value
        / result.point_estimate.enterprise_value
        if result.point_estimate.enterprise_value
        else float("nan")
    )

    lines = [
        "=" * 76,
        f"{result.ticker} | Professional Monte Carlo DCF",
        "=" * 76,
        f"Current price                    {_money_or_na(result.current_price)}",
        f"Deterministic DCF                {_money_or_na(result.point_estimate.intrinsic_value_per_share)}",
        f"Monte Carlo P10 / P50 / P90      {_money_or_na(sim.percentile_10)} / "
        f"{_money_or_na(sim.median)} / {_money_or_na(sim.percentile_90)}",
        f"Simulation mean / std. dev.      {_money_or_na(sim.mean)} / "
        f"{_money_or_na(sim.standard_deviation)}",
        f"Valid / rejected scenarios       {sim.valid_scenarios:,} / "
        f"{sim.rejected_scenarios:,}",
        "-" * 76,
        f"Financial data source            {result.data_source}",
        f"Latest financial period          {financials.latest_period}",
        f"Selected FCFF                    {_money_or_na(financials.selected_fcff, billions=True)}",
        f"FCFF selection                   {financials.selected_fcff_method}",
        f"Latest cash-flow FCFF            {_money_or_na(financials.latest_cash_fcff, billions=True)}",
        f"Normalized cash-flow FCFF        {_money_or_na(financials.normalized_fcff, billions=True)}",
        f"Latest bottom-up FCFF            {_money_or_na(financials.latest_bottom_up_fcff, billions=True)}",
        f"Net debt                         {_money_or_na(financials.net_debt, billions=True)}",
        f"Revenue / cash-FCFF CAGR         {_percent_or_na(financials.revenue_cagr)} / "
        f"{_percent_or_na(financials.cash_fcff_cagr)}",
        "-" * 76,
        f"Explicit growth                  {result.growth.rate:.2%}",
        f"Growth source                    {result.growth.source}",
        f"Terminal growth                  {result.terminal_growth_rate:.2%}",
        f"Explicit forecast period         {result.forecast_years} years",
    ]
    lines.extend(
        [
            f"Beta / R-squared                 {risk.beta:.2f} / {_percent_or_na(risk.r_squared)}",
            f"Beta source                      {risk.source}",
            f"Risk-free rate                   {costs.risk_free_rate:.2%} ({costs.risk_free_source})",
            f"Cost of equity / debt            {costs.cost_of_equity:.2%} / "
            f"{costs.pre_tax_cost_of_debt:.2%}",
            f"WACC raw / applied               {costs.wacc_before_floor:.2%} / {costs.wacc:.2%}",
            f"WACC source                      {costs.wacc_source}",
            f"Terminal value / enterprise      {terminal_share:.1%}",
            f"Reverse-DCF implied growth       {_percent_or_na(result.reverse_dcf_growth)}",
            "-" * 76,
            "Sensitivity: intrinsic value per share",
            "Growth \\ WACC" + "".join(
                f"{wacc:>13.2%}" for wacc in result.sensitivity_waccs
            ),
        ]
    )
    for label, values in result.sensitivity:
        lines.append(
            f"{label:<13}"
            + "".join(f"{'$' + format(value, ',.2f'):>13}" for value in values)
        )
    if result.warnings:
        lines.extend(["-" * 76, "Model and data warnings:"])
        lines.extend(f"  - {warning}" for warning in result.warnings)
    lines.extend(
        [
            "-" * 76,
            "Research use only. Review source filings and analyst assumptions before use.",
        ]
    )
    return "\n".join(lines)


def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelInputError(f"configuration file not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ModelInputError(f"invalid JSON configuration: {error}") from error
    if not isinstance(data, dict):
        raise ModelInputError("configuration JSON must contain one object")
    valid = {field.name for field in fields(ValuationConfig)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ModelInputError(f"unknown configuration field(s): {', '.join(unknown)}")
    tuple_fields = {
        "auto_growth_bounds",
        "growth_bounds",
        "terminal_growth_bounds",
        "wacc_bounds",
    }
    for name in tuple_fields:
        if name in data:
            data[name] = tuple(data[name])
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a configurable, normalized Monte Carlo DCF valuation."
    )
    parser.add_argument("ticker", nargs="?", help="Yahoo Finance ticker, e.g. MDLZ")
    parser.add_argument("--config", help="JSON file with ValuationConfig fields")
    parser.add_argument("--growth", type=float, help="five-year growth, e.g. 0.025")
    parser.add_argument("--terminal-growth", type=float, help="perpetual growth rate")
    parser.add_argument("--wacc", type=float, help="manual WACC override")
    parser.add_argument("--risk-free-rate", type=float, help="manual risk-free rate")
    parser.add_argument("--erp", type=float, help="equity risk premium")
    parser.add_argument("--beta", type=float, help="manual beta override")
    parser.add_argument(
        "--fcff-method", choices=("normalized", "latest", "bottom_up")
    )
    parser.add_argument(
        "--fcff-override-bn",
        type=float,
        help="manual FCFF override in billions of reporting currency",
    )
    parser.add_argument("--normalization-years", type=int)
    parser.add_argument("--market", help="market benchmark ticker")
    parser.add_argument(
        "--data-source",
        choices=("auto", "yahoo", "sec"),
        help="financial statement source",
    )
    parser.add_argument(
        "--sec-user-agent",
        help="SEC-required identity, e.g. 'Your Name your.email@example.com'",
    )
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-report", help="optional text report output path")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ValuationConfig:
    data: dict[str, Any] = load_config_file(args.config) if args.config else {}
    if args.ticker:
        data["ticker"] = args.ticker
    elif "ticker" not in data:
        if not sys.stdin.isatty():
            raise ModelInputError("ticker is required")
        data["ticker"] = input("Enter a ticker symbol (e.g., MDLZ): ").strip()

    overrides = {
        "explicit_growth_rate": args.growth,
        "terminal_growth_rate": args.terminal_growth,
        "wacc_override": args.wacc,
        "risk_free_rate_override": args.risk_free_rate,
        "equity_risk_premium": args.erp,
        "beta_override": args.beta,
        "fcff_method": args.fcff_method,
        "normalization_years": args.normalization_years,
        "market_symbol": args.market,
        "data_source": args.data_source,
        "sec_user_agent": args.sec_user_agent,
        "simulations": args.simulations,
        "random_seed": args.seed,
    }
    for name, value in overrides.items():
        if value is not None:
            data[name] = value
    if args.fcff_override_bn is not None:
        data["fcff_override"] = args.fcff_override_bn * 1e9
    return ValuationConfig(**data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )
    try:
        config = config_from_args(args)
        if args.show_config:
            print(json.dumps(asdict(config), indent=2, ensure_ascii=False))
            return 0
        result = MonteCarloDCFEngine(config).run()
        report = format_report(result)
        print(report)
        if args.save_report:
            output_path = Path(args.save_report)
            output_path.write_text(report + "\n", encoding="utf-8")
            print(f"\nReport saved to: {output_path.resolve()}")
        return 0
    except ValuationError as error:
        LOGGER.error("Valuation failed: %s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Valuation cancelled by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
