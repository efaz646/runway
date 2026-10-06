"""Bill checker and supplier emails.

Code totals each supplier's bills by month, works out the usual level and finds any
spike. Gemini suggests what to do about the spike (in the agent's plan, see llm.py)
and writes the query email. It also writes the request to move a payment when the
agent chooses "delay one payment". Every number in a draft is checked against the
data, with a template as the fallback. Nothing is ever sent.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

import config
import llm
from forecast import Forecast, money, short_date, usual_level

REF_PATTERN = re.compile(r"\b(?:REF|ACC|POL)\s+([\w-]+)")


def monthly_supplier_costs(statement: pd.DataFrame, as_of: date = config.AS_OF_DATE) -> pd.DataFrame:
    """One row per supplier per month: category, supplier, bill, month_start, month, amount, usual, is_spike."""
    bills = statement[(statement["category"].isin(config.SUPPLIER_BILL_CATEGORIES)) & (statement["date"] < as_of)].copy()
    bills["month_start"] = bills["date"].map(lambda d: d.replace(day=1))
    monthly = bills.groupby(["category", "month_start"], as_index=False)["amount"].sum()
    monthly["amount"] = monthly["amount"].abs().round(2)
    parts = []
    for category, group in monthly.groupby("category", sort=False):
        group = group.sort_values("month_start").copy()
        usual, flags = usual_level(group["amount"].tolist())
        group["usual"] = usual
        group["is_spike"] = flags
        parts.append(group)
    out = pd.concat(parts, ignore_index=True)
    out["supplier"] = out["category"].map(lambda c: config.SUPPLIERS[c]["name"])
    out["bill"] = out["category"].map(lambda c: config.SUPPLIERS[c]["bill"])
    out["month"] = out["month_start"].map(lambda d: f"{d:%b %Y}")
    order = {c: i for i, c in enumerate(config.SUPPLIER_BILL_CATEGORIES)}
    return out.sort_values(["category", "month_start"], key=lambda s: s.map(order) if s.name == "category" else s).reset_index(drop=True)


def account_ref(statement: pd.DataFrame, category: str) -> str | None:
    """The account reference printed on the supplier's bank lines, e.g. 88213."""
    for description in reversed(statement.loc[statement["category"] == category, "description"].tolist()):
        match = REF_PATTERN.search(description)
        if match:
            return match.group(1)
    return None


def find_spikes(statement: pd.DataFrame) -> list[dict]:
    monthly = monthly_supplier_costs(statement)
    spikes = []
    for r in monthly[monthly["is_spike"]].itertuples():
        spikes.append(
            {
                "category": r.category,
                "supplier": r.supplier,
                "bill": r.bill,
                "month": r.month,
                "month_name": f"{r.month_start:%B}",
                "amount": round(float(r.amount), 2),
                "usual": round(float(r.usual), 2),
                "excess": round(float(r.amount - r.usual), 2),
                "times_usual": round(float(r.amount / r.usual), 1),
                "account_ref": account_ref(statement, r.category),
            }
        )
    return spikes


def bill_facts(statement: pd.DataFrame) -> dict:
    """What the agent may say about supplier bills, already calculated and formatted."""
    monthly = monthly_supplier_costs(statement)
    spikes = find_spikes(statement)
    return {
        "suppliers": [
            {
                "supplier": group["supplier"].iloc[0],
                "bill": group["bill"].iloc[0],
                "usual_monthly_amount": money(group["usual"].iloc[0]),
                "months": [{"month": r.month, "amount": money(r.amount)} for r in group.itertuples()],
            }
            for _, group in monthly.groupby("category", sort=False)
        ],
        "spike_count": len(spikes),
        "spikes": [
            {
                "supplier": s["supplier"],
                "bill": s["bill"],
                "month": s["month"],
                "amount": money(s["amount"]),
                "usual_monthly_amount": money(s["usual"]),
                "more_than_usual": money(s["excess"]),
                "times_usual": s["times_usual"],
                "account_ref": s["account_ref"],
            }
            for s in spikes
        ],
    }


def fallback_suggestions(spike: dict) -> list[str]:
    """Used only when Gemini is unavailable."""
    return [
        f"Ask {spike['supplier']} whether the {spike['month_name']} bill used an estimated meter reading.",
        "Take a meter reading today and send it with your query.",
        "Ask for a corrected bill or a breakdown of the charges and the tariff used.",
    ]


# --------------------------------------------------------------- query email


def query_facts(spike: dict) -> dict:
    return {
        "from": f"{config.OWNER_NAME}, {config.BUSINESS_NAME}",
        "supplier": spike["supplier"],
        "account_ref": spike["account_ref"],
        "bill": spike["bill"],
        "month": spike["month"],
        "amount_charged": money(spike["amount"]),
        "usual_monthly_amount": money(spike["usual"]),
        "more_than_usual": money(spike["excess"]),
        "times_usual": spike["times_usual"],
    }


def query_template(spike: dict) -> tuple[str, str]:
    ref = f" (account {spike['account_ref']})" if spike["account_ref"] else ""
    subject = f"Query: {spike['month']} {spike['bill'].lower()} bill{ref}"
    body = (
        f"Hello,\n\nOur {spike['bill'].lower()} bill for {spike['month']}{ref} was {money(spike['amount'])}. "
        f"Our usual monthly bill is {money(spike['usual'])}, so this is {money(spike['excess'])} more than usual.\n\n"
        "Please could you check whether this bill is based on an estimated meter reading, confirm the readings and "
        "tariff used, and send us a corrected bill or a full breakdown? We can send a meter reading if that helps.\n\n"
        f"Thanks,\n{config.OWNER_NAME}\n{config.BUSINESS_NAME}"
    )
    return subject, body


def draft_query_email(spike: dict, use_gemini: bool = True) -> llm.Draft:
    contact = config.SUPPLIERS[spike["category"]]
    brief = (f"Write an email to {spike['supplier']} querying the {spike['month']} {spike['bill'].lower()} bill. "
             "Quote the account reference, the amount charged and the usual monthly amount, ask whether it was based on an "
             "estimated reading, and ask for a check and a corrected bill or breakdown. Polite and to the point.")
    required = [spike["supplier"]] + ([spike["account_ref"]] if spike["account_ref"] else [])
    return llm.write_draft(
        kind="query_bill",
        label=f"{spike['supplier']} ({spike['month']})",
        to=f"{spike['supplier']} <{contact['email']}>",
        brief=brief,
        facts=query_facts(spike),
        required_numbers=[spike["amount"], spike["usual"]],
        required_phrases=required,
        fallback=lambda: query_template(spike),
        use_gemini=use_gemini,
    )


# --------------------------------------------------------------- delay request


def delay_facts(fc: Forecast, statement: pd.DataFrame) -> dict:
    c = fc.delay_candidate
    contact = config.SUPPLIERS[c["category"]]
    return {
        "from": f"{config.OWNER_NAME}, {config.BUSINESS_NAME}",
        "supplier": contact["name"],
        "account_ref": account_ref(statement, c["category"]),
        "payment": contact["bill"],
        "usual_amount": money(c["amount"]),
        "due_date": short_date(c["date"]),
        "ask_to_move_to": short_date(c["new_date"]),
        "days_later": config.DELAY_DAYS,
    }


def delay_template(facts: dict) -> tuple[str, str]:
    ref = f" (account {facts['account_ref']})" if facts["account_ref"] else ""
    subject = f"Request to move our {facts['due_date']} payment{ref}"
    body = (
        f"Hello,\n\nOur monthly {facts['payment'].lower()} payment{ref}, usually around {facts['usual_amount']}, "
        f"is due on {facts['due_date']}. Could we move it to {facts['ask_to_move_to']} this once? "
        "We have a few large bills landing in the same week and want to keep everything paid on time.\n\n"
        f"We value the account and will pay in full on the new date. Please let me know if that's OK.\n\n"
        f"Thanks,\n{config.OWNER_NAME}\n{config.BUSINESS_NAME}"
    )
    return subject, body


def draft_delay_email(fc: Forecast, statement: pd.DataFrame, use_gemini: bool = True) -> llm.Draft:
    facts = delay_facts(fc, statement)
    contact = config.SUPPLIERS[fc.delay_candidate["category"]]
    brief = (f"Write a short, polite email to {facts['supplier']} asking to move the payment due on {facts['due_date']} "
             f"to {facts['ask_to_move_to']}, this once. Quote the account reference. Say it is because several large bills "
             "fall in the same week, and that it will be paid in full on the new date. Keep the relationship warm.")
    required = [facts["supplier"]] + ([facts["account_ref"]] if facts["account_ref"] else [])
    return llm.write_draft(
        kind="delay_payment",
        label=f"{facts['supplier']} ({facts['due_date']} payment)",
        to=f"{facts['supplier']} <{contact['email']}>",
        brief=brief,
        facts=facts,
        required_numbers=[],
        required_phrases=required,
        fallback=lambda: delay_template(facts),
        use_gemini=use_gemini,
    )
