"""Late payment chaser.

Code finds the overdue invoices, works out how many days overdue each one is and
picks the tone. Gemini writes each reminder. llm.write_draft then checks every
number in the draft against that invoice's own facts, and uses a template if the
check fails. Nothing is ever sent.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

import config
import llm
from forecast import Forecast, first_short_week, money, short_date, with_changes

TONE_GUIDE = {
    "friendly": "Friendly and light. Assume it is an oversight and thank them.",
    "firm": "Polite but firm. It is now well overdue, so ask them to pay or to say if there is a problem with the invoice.",
    "final": "Final reminder. Clear and serious but professional. Say that new work will be put on hold until the account is settled.",
}


def tone_for(days_overdue: int) -> str:
    if days_overdue <= config.TONE_FRIENDLY_MAX_DAYS:
        return "friendly"
    if days_overdue <= config.TONE_FIRM_MAX_DAYS:
        return "firm"
    return "final"


def unpaid_invoices(invoices: pd.DataFrame, as_of: date = config.AS_OF_DATE) -> pd.DataFrame:
    """Unpaid invoices with days overdue (negative = not yet due), overdue flag and tone. Oldest first."""
    unpaid = invoices[~invoices["paid"]].copy()
    unpaid["days_overdue"] = unpaid["due_date"].map(lambda d: (as_of - d).days)
    unpaid["overdue"] = unpaid["days_overdue"] > 0
    unpaid["tone"] = unpaid["days_overdue"].map(lambda d: tone_for(d) if d > 0 else "")
    return unpaid.sort_values("days_overdue", ascending=False).reset_index(drop=True)


def overdue_invoices(invoices: pd.DataFrame, as_of: date = config.AS_OF_DATE) -> pd.DataFrame:
    unpaid = unpaid_invoices(invoices, as_of)
    return unpaid[unpaid["overdue"]].reset_index(drop=True)


def invoice_facts(invoices: pd.DataFrame, fc: Forecast) -> dict:
    """What the agent may say about money owed, already calculated and formatted."""
    unpaid = unpaid_invoices(invoices, fc.as_of)
    overdue = unpaid[unpaid["overdue"]]
    upcoming = unpaid[~unpaid["overdue"]]
    overdue_total = round(float(overdue["amount"].sum()), 2)
    facts = {
        "overdue_count": len(overdue),
        "overdue_total": money(overdue_total),
        "overdue_invoices": [
            {
                "invoice_number": r.invoice_number,
                "customer": r.customer,
                "contact_name": r.contact_name,
                "amount": money(r.amount),
                "due_date": short_date(r.due_date),
                "days_overdue": int(r.days_overdue),
                "reminder_tone": r.tone,
            }
            for r in overdue.itertuples()
        ],
        "not_yet_due": [
            {
                "invoice_number": r.invoice_number,
                "customer": r.customer,
                "amount": money(r.amount),
                "due_date": short_date(r.due_date),
                "due_in_days": int(-r.days_overdue),
            }
            for r in upcoming.itertuples()
        ],
        "customers_who_owe_money": int(unpaid["customer"].nunique()),
        "total_owed": money(unpaid["amount"].sum()),
    }
    if overdue_total > 0:
        after = with_changes(fc, [(fc.as_of, overdue_total)])
        short = first_short_week(after)
        low = after.loc[after["closing_balance"].idxmin()]
        facts["if_overdue_invoices_paid_this_week"] = {
            "first_week_short": short,
            "cash_stays_above_zero_for_all_weeks": short is None,
            "lowest_balance": money(low["closing_balance"]),
            "lowest_balance_week": int(low["week"]),
        }
    return facts


# ---------------------------------------------------------------------- drafts


def reminder_facts(row) -> dict:
    return {
        "from": f"{config.OWNER_NAME}, {config.BUSINESS_NAME}",
        "customer": row.customer,
        "contact_name": row.contact_name,
        "invoice_number": row.invoice_number,
        "amount": money(row.amount),
        "due_date": short_date(row.due_date),
        "days_overdue": int(row.days_overdue),
        "tone": row.tone,
        "ask_them_to_pay_within_days": config.REMINDER_PAY_WITHIN_DAYS,
    }


def reminder_template(row) -> tuple[str, str]:
    """Plain fallback reminder, used when Gemini is unavailable or its draft fails the check."""
    hello = f"Hi {row.contact_name}," if row.contact_name else "Hello,"
    amount, due, days = money(row.amount), short_date(row.due_date), int(row.days_overdue)
    within = config.REMINDER_PAY_WITHIN_DAYS
    sign_off = f"Thanks,\n{config.OWNER_NAME}\n{config.BUSINESS_NAME}"
    if row.tone == "friendly":
        subject = f"Friendly reminder: invoice {row.invoice_number} ({amount})"
        text = (f"Just a quick reminder that invoice {row.invoice_number} for {amount} to {row.customer} was due on {due} "
                f"and is now {days} days overdue. If it's already on its way, please ignore this. "
                f"Otherwise, could you arrange payment within the next {within} days?")
    elif row.tone == "firm":
        subject = f"Overdue invoice {row.invoice_number} ({amount}), {days} days late"
        text = (f"Invoice {row.invoice_number} for {amount} to {row.customer} was due on {due} and is now {days} days overdue. "
                f"Please arrange payment within {within} days, or let me know if there is a problem with the invoice.")
    else:
        subject = f"Final reminder: invoice {row.invoice_number} ({amount}), {days} days overdue"
        text = (f"Invoice {row.invoice_number} for {amount} to {row.customer} was due on {due} and is now {days} days overdue, "
                f"despite earlier reminders. Please pay within {within} days. Until the account is settled, "
                f"we'll have to put new work on hold.")
    return subject, f"{hello}\n\n{text}\n\n{sign_off}"


def draft_reminder(row, use_gemini: bool = True) -> llm.Draft:
    to = f"{row.contact_name}, {row.customer} <{row.email}>" if row.email else row.customer
    brief = (f"Write a payment reminder email to {row.contact_name or 'the accounts team'} at {row.customer}. "
             f"Tone: {TONE_GUIDE[row.tone]} Mention {row.customer} by name, the invoice number, the amount, "
             f"the due date and how many days overdue it is, and ask them to pay within the number of days given.")
    return llm.write_draft(
        kind="chase_overdue_invoices",
        label=f"{row.customer} ({row.tone})",
        to=to,
        brief=brief,
        facts=reminder_facts(row),
        required_numbers=[row.amount, int(row.days_overdue)],
        required_phrases=[row.customer, row.invoice_number],
        fallback=lambda: reminder_template(row),
        use_gemini=use_gemini,
    )
