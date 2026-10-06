"""A stand-in for the Gemini API, used by the automated tests.

It behaves like a sensible Gemini (looks up the tools, submits a plan, writes drafts
from the facts it is given) and can be told to make the mistakes Runway must catch:
an invented number in the plan, or an invented phone number in a draft. It never
touches the network. Live behaviour is covered by tests/test_live.py.
"""

from __future__ import annotations

import json

from google.genai import types


def _response(parts: list[types.Part]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


def _text(text: str) -> types.GenerateContentResponse:
    return _response([types.Part(text=text)])


def _calls(*calls: tuple[str, dict]) -> types.GenerateContentResponse:
    return _response([types.Part(function_call=types.FunctionCall(id=f"call{i}", name=n, args=a)) for i, (n, a) in enumerate(calls)])


class FakeGemini:
    def __init__(self, plan_mistake: bool = False, draft_mistake_for: set[str] | None = None, fail: bool = False,
                 always_bad_drafts: bool = False, skip_customer: str | None = None):
        self.plan_mistake = plan_mistake
        self.draft_mistake_for = draft_mistake_for or set()
        self.always_bad_drafts = always_bad_drafts
        self.skip_customer = skip_customer  # leave one customer out of "who owes me money?" (Runway must catch it)
        self.fail = fail
        self.requests: list = []
        self.models = self  # so client.models.generate_content works

    # ------------------------------------------------------------------ API
    def generate_content(self, model, contents, config):
        self.requests.append((model, contents, config))
        if self.fail:
            raise ConnectionError("simulated Gemini outage")
        if config.tools:
            return self._agent_turn(contents)
        schema = config.response_json_schema or {}
        if "text" in schema.get("properties", {}):
            return self._spoken(contents)
        if not schema:
            return _text("OK")
        return self._draft(contents)

    # ---------------------------------------------------------------- agent
    def _agent_turn(self, contents) -> types.GenerateContentResponse:
        results, submits = {}, 0
        for content in contents:
            for part in content.parts or []:
                fr = part.function_response
                if fr is None:
                    continue
                if fr.name == "submit_plan":
                    submits += 1
                else:
                    results[fr.name] = fr.response["result"]
        if not results:
            return _calls(("get_cash_forecast", {}), ("get_unpaid_invoices", {}), ("get_supplier_bills", {}))
        plan = self._plan(results)
        if self.plan_mistake and submits == 0:
            plan["headline"] += " That is about £5,600 short."
        return _calls(("submit_plan", plan))

    @staticmethod
    def _plan(r: dict) -> dict:
        f, inv, b = r["get_cash_forecast"], r["get_unpaid_invoices"], r["get_supplier_bills"]
        spike, move = b["spikes"][0], f["movable_payment"]
        return {
            "headline": (f"Cash runs short in week {f['runs_short_in_week']}, when the balance reaches "
                         f"{f['balance_at_end_of_shortfall_week']}. Chasing the {inv['overdue_total']} of overdue invoices would cover it."),
            "actions": [
                {"priority": 1, "type": "chase_overdue_invoices", "title": "Chase the overdue invoices",
                 "why": f"{inv['overdue_count']} customers owe {inv['overdue_total']}; paid this week, cash stays above zero.",
                 "invoice_numbers": [i["invoice_number"] for i in inv["overdue_invoices"]]},
                {"priority": 2, "type": "query_bill", "title": f"Query the {spike['month']} electricity bill",
                 "why": f"{spike['supplier']} charged {spike['amount']}, against a usual {spike['usual_monthly_amount']}.",
                 "supplier": spike["supplier"]},
                {"priority": 3, "type": "delay_payment", "title": "Ask to move the parts payment",
                 "why": f"Moving it from {move['due']} to {move['could_move_to']} leaves {move['balance_at_end_of_shortfall_week_if_moved']} in week {f['runs_short_in_week']}.",
                 "supplier": "Pennine Motor Factors"},
            ],
            "bill_suggestions": [
                f"Ask {spike['supplier']} if the {spike['month']} bill used an estimated meter reading.",
                "Take a meter reading today and send it with your query.",
                "Ask for a corrected bill or a breakdown of the tariff.",
            ],
        }

    # --------------------------------------------------------------- drafts
    def _draft(self, prompt: str) -> types.GenerateContentResponse:
        facts = json.loads(prompt.split("Facts (JSON):\n", 1)[1].split("\n\nYour last draft", 1)[0])
        retry = "Your last draft" in prompt
        if "invoice_number" in facts:
            subject = f"Invoice {facts['invoice_number']} for {facts['customer']}"
            body = (f"Hi {facts['contact_name']},\n\nInvoice {facts['invoice_number']} for {facts['amount']} was due on "
                    f"{facts['due_date']} and is now {facts['days_overdue']} days overdue. Please pay within "
                    f"{facts['ask_them_to_pay_within_days']} days.\n\nThanks,\n{facts['from']}")
            if (facts["customer"] in self.draft_mistake_for and not retry) or self.always_bad_drafts:
                body += "\nCall me on 01274 555123."
        elif "amount_charged" in facts:
            subject = f"Query on account {facts['account_ref']}"
            body = (f"Hello {facts['supplier']},\n\nOur {facts['month']} bill was {facts['amount_charged']}, against a usual "
                    f"{facts['usual_monthly_amount']}. Was it an estimated reading? Please check it.\n\n{facts['from']}")
        else:
            subject = f"Moving our {facts['due_date']} payment"
            body = (f"Hello {facts['supplier']},\n\nCould we move our account {facts['account_ref']} payment from "
                    f"{facts['due_date']} to {facts['ask_to_move_to']}?\n\n{facts['from']}")
        return _text(json.dumps({"subject": subject, "body": body}))

    # ------------------------------------------------- briefing and answers
    def _spoken(self, prompt: str) -> types.GenerateContentResponse:
        facts = json.loads(prompt.split("Facts (JSON):\n", 1)[1].split("\n\nYour last draft", 1)[0])
        if "question" in facts:
            owed = facts["money_owed"]
            if "owe" in facts["question"].lower():
                rows = owed["overdue_invoices"] + owed["not_yet_due"]
                text = f"{owed['customers_who_owe_money']} customers owe you {owed['total_owed']}. " + " ".join(
                    f"{r['customer']} owes {r['amount']}." for r in rows)
                if self.skip_customer:
                    text = " ".join(part for part in text.split(". ") if self.skip_customer not in part)
            else:
                text = f"Cash runs short in week {facts['forecast']['runs_short_in_week']}."
        else:
            text = (f"Good morning {facts['owner']}. You have {facts['cash_today']}. Cash runs short in week "
                    f"{facts['runs_short_in_week']}, the week of {facts['week_starting']}, and you'd be {facts['short_by']} short. "
                    f"{facts['overdue_invoices']} customers owe {facts['overdue_total']}. Your drafts are ready. Nothing has been sent.")
        return _text(json.dumps({"text": text}))
