"""Cash runway forecast.

Cleans Dave's messy bank statement, projects the next 8 weeks of cash in and out,
and finds how many weeks of cash are left and the first week the balance goes
below zero. Every figure is calculated here, in code, from the data. Gemini never
produces a number.

How the projection works (all from the statement and invoice list):
  * Current balance: sum of every statement line, including the balance brought forward.
  * Regular payments (wages, rent, bills, VAT): next dates follow the last payment's
    pattern (weekly, monthly or quarterly); the amount is the usual level of recent
    payments (median, ignoring spikes), so a one-off spike is not repeated.
  * Card takings and other small spending: weekly average of the last 8 full weeks.
  * Invoices: unpaid invoices that are not yet due are expected on their due date.
    Overdue invoices are NOT counted, because they are already late; chasing them is
    one of the actions.
  * Weekly closing balance = current balance + all money in and out up to that week's end.
"""

from __future__ import annotations

import calendar
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

import config

# ------------------------------------------------------------------ formatting


def money(value: float) -> str:
    """£1,234.56 or -£1,234.56. The single formatter for every amount Runway shows or checks."""
    value = round(float(value) + 0.0, 2)
    sign = "-" if value < 0 else ""
    return f"{sign}£{abs(value):,.2f}"


def short_date(d: date) -> str:
    """5 Oct 2026."""
    return f"{d.day} {d:%b %Y}"


# --------------------------------------------------------------------- parsing

DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d-%m-%y", "%d/%m/%y")


def parse_date(text) -> date:
    text = " ".join(str(text).split())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {text!r}")


def parse_amount(text) -> float:
    """Handles £1,234.50, +£1,234.50, -1234.50, 1,234.50 DR and (1,234.50)."""
    s = str(text).strip().replace("£", "").replace(",", "").replace(" ", "").upper()
    negative = False
    if s.endswith("DR"):
        negative, s = True, s[:-2]
    elif s.endswith("CR"):
        s = s[:-2]
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    if s.startswith("-"):
        negative, s = True, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    value = float(s)
    return -value if negative else value


def categorise(description: str) -> str:
    text = " ".join(description.upper().split()) + " "
    for category, keywords in config.CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return config.DEFAULT_CATEGORY


def load_bank_statement(path: Path = config.BANK_STATEMENT_CSV) -> pd.DataFrame:
    """Return a clean statement: date, description, amount, category (oldest first)."""
    raw = pd.read_csv(path, dtype=str, skip_blank_lines=True, encoding="utf-8-sig")
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw = raw.dropna(how="all")
    clean = pd.DataFrame(
        {
            "date": raw["date"].map(parse_date),
            "description": raw["description"].map(lambda s: " ".join(str(s).upper().split())),
            "amount": raw["amount"].map(parse_amount),
        }
    )
    clean["category"] = clean["description"].map(categorise)
    return clean.sort_values("date", kind="stable").reset_index(drop=True)


def load_invoices(path: Path = config.INVOICES_CSV) -> pd.DataFrame:
    """Return invoices: customer, contact_name, email, invoice_number, amount, due_date, paid (bool)."""
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig").dropna(how="all")
    raw.columns = [c.strip().lower() for c in raw.columns]
    blank = pd.Series([""] * len(raw), index=raw.index)
    return pd.DataFrame(
        {
            "customer": raw["customer"].str.strip(),
            "contact_name": raw.get("contact_name", blank).fillna("").str.strip(),
            "email": raw.get("email", blank).fillna("").str.strip(),
            "invoice_number": raw["invoice_number"].str.strip(),
            "amount": raw["amount"].map(parse_amount),
            "due_date": raw["due_date"].map(parse_date),
            "paid": raw["paid"].str.strip().str.lower().isin(["yes", "y", "true", "paid"]),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------- usual levels


def usual_level(values: list[float]) -> tuple[float, list[bool]]:
    """Return (usual level, spike flags) for a list of positive amounts.

    A value is a spike if it is more than config.SPIKE_RATIO times the median of the
    other values (needs at least 4 values). The usual level is the median of the
    values that are not spikes. Used by both the forecast and the bill checker, so
    "usual" means the same thing everywhere.
    """
    values = [round(abs(v), 2) for v in values]
    flags = [False] * len(values)
    if len(values) >= 4:
        for i, v in enumerate(values):
            others = values[:i] + values[i + 1:]
            flags[i] = v > config.SPIKE_RATIO * statistics.median(others)
    normal = [v for v, spike in zip(values, flags) if not spike] or values
    return round(statistics.median(normal), 2), flags


# ------------------------------------------------------------------ projection


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    year, month = d.year + m // 12, m % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _future_dates(last: date, cadence: str, start: date, end: date) -> list[date]:
    out, i = [], 1
    while True:
        if cadence == "weekly":
            d = last + timedelta(weeks=i)
        elif cadence == "monthly":
            d = add_months(last, i)
        elif cadence == "quarterly":
            d = add_months(last, 3 * i)
        else:
            raise ValueError(cadence)
        if d > end:
            return out
        if d >= start:
            out.append(d)
        i += 1


def week_of(d: date, as_of: date) -> int:
    """Week 1 is the 7 days starting on the as-of date."""
    return (d - as_of).days // 7 + 1


def weekly_balances(start_balance: float, flows: pd.DataFrame, as_of: date, n_weeks: int) -> pd.DataFrame:
    weeks, balance = [], start_balance
    for w in range(1, n_weeks + 1):
        in_week = flows.loc[flows["week"] == w, "amount"]
        money_in = round(float(in_week[in_week > 0].sum()), 2)
        money_out = round(float(in_week[in_week < 0].sum()), 2)
        balance = round(balance + money_in + money_out, 2)
        start = as_of + timedelta(weeks=w - 1)
        weeks.append(
            {
                "week": w,
                "week_start": start,
                "week_label": f"Wk {w} ({start.day} {start:%b})",
                "money_in": money_in,
                "money_out": money_out,
                "closing_balance": balance,
            }
        )
    return pd.DataFrame(weeks)


def first_short_week(weeks: pd.DataFrame) -> int | None:
    short = weeks[weeks["closing_balance"] < 0]
    return int(short["week"].iloc[0]) if not short.empty else None


@dataclass
class Forecast:
    as_of: date
    current_balance: float
    weeks: pd.DataFrame  # week, week_start, week_label, money_in, money_out, closing_balance
    flows: pd.DataFrame  # date, week, category, label, amount (signed)
    shortfall_week: int | None
    shortfall_week_start: date | None
    shortfall_balance: float | None  # closing balance in the shortfall week
    weeks_of_cash_left: int  # full weeks before the shortfall (or the whole horizon)
    lowest_balance: float
    lowest_week: int
    delay_candidate: dict | None  # largest movable payment in the shortfall week

    @property
    def warning(self) -> str:
        if self.shortfall_week is None:
            return f"Cash stays above zero for the next {len(self.weeks)} weeks."
        return f"Cash runs short in week {self.shortfall_week}"

    @property
    def runs_short(self) -> bool:
        return self.shortfall_week is not None


def build_forecast(
    statement: pd.DataFrame,
    invoices: pd.DataFrame,
    as_of: date = config.AS_OF_DATE,
    n_weeks: int = config.FORECAST_WEEKS,
) -> Forecast:
    history = statement[statement["date"] < as_of]
    current_balance = round(float(history["amount"].sum()), 2)
    horizon_end = as_of + timedelta(weeks=n_weeks) - timedelta(days=1)
    rows: list[dict] = []

    # Regular payments, projected from their own history at their usual level.
    for category, cadence in config.RECURRING_CADENCE.items():
        past = history[history["category"] == category].sort_values("date")
        if past.empty:
            continue
        recent = past.tail(config.RECURRING_LOOKBACK[cadence])
        usual, _ = usual_level(recent["amount"].tolist())
        sign = -1 if recent["amount"].sum() < 0 else 1
        for d in _future_dates(past["date"].iloc[-1], cadence, as_of, horizon_end):
            rows.append({"date": d, "category": category, "label": config.CATEGORY_LABELS[category], "amount": sign * usual})

    # Card takings and small spending: weekly average of the last 8 full weeks.
    window_start = as_of - timedelta(weeks=config.AVERAGE_WINDOW_WEEKS)
    window = history[history["date"] >= window_start]
    for category in config.WEEKLY_AVERAGE_CATEGORIES:
        weekly = round(float(window.loc[window["category"] == category, "amount"].sum()) / config.AVERAGE_WINDOW_WEEKS, 2)
        for w in range(n_weeks):
            rows.append(
                {
                    "date": as_of + timedelta(weeks=w),
                    "category": category,
                    "label": f"{config.CATEGORY_LABELS[category]} (weekly average)",
                    "amount": weekly,
                }
            )

    # Unpaid invoices that are not yet due arrive on their due date. Overdue ones are not counted.
    due = invoices[(~invoices["paid"]) & (invoices["due_date"] >= as_of) & (invoices["due_date"] <= horizon_end)]
    for inv in due.itertuples():
        rows.append(
            {
                "date": inv.due_date,
                "category": "invoice_due",
                "label": f"Invoice {inv.invoice_number} from {inv.customer}",
                "amount": round(float(inv.amount), 2),
            }
        )

    flows = pd.DataFrame(rows)
    flows["week"] = flows["date"].map(lambda d: week_of(d, as_of))
    flows = flows.sort_values(["date", "category"]).reset_index(drop=True)
    weeks_df = weekly_balances(current_balance, flows, as_of, n_weeks)

    shortfall_week = first_short_week(weeks_df)
    lowest_row = weeks_df.loc[weeks_df["closing_balance"].idxmin()]

    delay_candidate = None
    if shortfall_week is not None:
        movable = flows[(flows["week"] == shortfall_week) & (flows["category"].isin(config.DEFERRABLE_CATEGORIES))]
        if not movable.empty:
            pick = movable.loc[movable["amount"].idxmin()]  # most negative = largest payment
            delay_candidate = {
                "label": pick["label"],
                "category": pick["category"],
                "amount": round(abs(float(pick["amount"])), 2),
                "date": pick["date"],
                "week": shortfall_week,
                "new_date": pick["date"] + timedelta(days=config.DELAY_DAYS),
            }

    return Forecast(
        as_of=as_of,
        current_balance=current_balance,
        weeks=weeks_df,
        flows=flows,
        shortfall_week=shortfall_week,
        shortfall_week_start=(as_of + timedelta(weeks=shortfall_week - 1)) if shortfall_week else None,
        shortfall_balance=float(weeks_df.loc[weeks_df["week"] == shortfall_week, "closing_balance"].iloc[0]) if shortfall_week else None,
        weeks_of_cash_left=(shortfall_week - 1) if shortfall_week else n_weeks,
        lowest_balance=float(lowest_row["closing_balance"]),
        lowest_week=int(lowest_row["week"]),
        delay_candidate=delay_candidate,
    )


def with_changes(fc: Forecast, changes: list[tuple[date, float]]) -> pd.DataFrame:
    """Weekly balances if extra money moves in (+) or out (-) on the given dates. Used for "what if" figures."""
    extra = pd.DataFrame([{"date": d, "amount": a, "week": week_of(d, fc.as_of)} for d, a in changes])
    flows = pd.concat([fc.flows[["date", "amount", "week"]], extra], ignore_index=True)
    return weekly_balances(fc.current_balance, flows, fc.as_of, len(fc.weeks))


def forecast_facts(fc: Forecast) -> dict:
    """Everything the agent may say about the forecast, already calculated and formatted."""
    facts = {
        "today": short_date(fc.as_of),
        "forecast_weeks": len(fc.weeks),
        "cash_today": money(fc.current_balance),
        "weeks_of_cash_left": fc.weeks_of_cash_left,
        "runs_short_in_week": fc.shortfall_week,
        "lowest_balance": money(fc.lowest_balance),
        "lowest_balance_week": fc.lowest_week,
        "weekly_closing_balances": [
            {"week": int(r.week), "week_starts": short_date(r.week_start), "closing_balance": money(r.closing_balance)}
            for r in fc.weeks.itertuples()
        ],
    }
    if fc.runs_short:
        facts["shortfall_week_starts"] = short_date(fc.shortfall_week_start)
        facts["balance_at_end_of_shortfall_week"] = money(fc.shortfall_balance)
        out = fc.flows[(fc.flows["week"] == fc.shortfall_week) & (fc.flows["amount"] < 0)].sort_values("amount")
        facts["payments_due_in_shortfall_week"] = [
            {"item": r.label, "date": short_date(r.date), "amount": money(abs(r.amount))} for r in out.itertuples()
        ]
    if fc.delay_candidate:
        c = fc.delay_candidate
        after = with_changes(fc, [(c["date"], c["amount"]), (c["new_date"], -c["amount"])])
        moved_short = first_short_week(after)
        facts["movable_payment"] = {
            "item": c["label"],
            "due": short_date(c["date"]),
            "usual_amount": money(c["amount"]),
            "could_move_to": short_date(c["new_date"]),
            "days_later": config.DELAY_DAYS,
            "balance_at_end_of_shortfall_week_if_moved": money(after.loc[after["week"] == fc.shortfall_week, "closing_balance"].iloc[0]),
            "first_week_short_if_moved": moved_short,
        }
    return facts


def load_and_forecast() -> tuple[pd.DataFrame, pd.DataFrame, Forecast]:
    statement = load_bank_statement()
    invoices = load_invoices()
    return statement, invoices, build_forecast(statement, invoices)


if __name__ == "__main__":
    import json

    _, _, fc = load_and_forecast()
    print(fc.weeks.to_string(index=False))
    print(fc.warning, "|", fc.weeks_of_cash_left, "weeks of cash left")
    print(json.dumps(forecast_facts(fc), indent=1))
