"""Gemini: the agent's brain, the drafts, and the number check.

Gemini decides and writes. Code calculates and checks.

  * analyse(): Gemini works as an agent. It calls Runway's tools to look up the facts
    (cash forecast, unpaid invoices, supplier bills), decides up to 3 prioritised
    actions plus suggestions for any bill spike, and submits its plan. Runway checks
    the plan. If it uses a number that is not in the data, the plan goes back to
    Gemini with the problem listed. Runway then has Gemini draft the email for every
    action it chose, in parallel.
  * write_draft(): every number in a draft must match the facts for that email, and
    the key figures must be present. One retry with the problems listed, then a
    plain template.
  * Fallbacks, so the demo never breaks: if Gemini fails, Runway uses the last
    verified analysis saved on this machine (cache/last_agent_run.json). If there is
    none, it uses a built-in plan with template drafts. Every figure is still
    calculated from the data.

Run `python llm.py` to make one test call with your GEMINI_API_KEY.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Callable

from google import genai
from google.genai import types

import config
from forecast import Forecast, forecast_facts, money, short_date


class GeminiUnavailable(Exception):
    """No key, or every model call failed. Callers fall back instead of crashing."""


# ======================================================================= client


def _client() -> genai.Client:
    key = config.gemini_api_key()
    if not key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set")
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=config.GEMINI_TIMEOUT_SECONDS * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),  # fail fast; Runway falls back instead
        ),
    )


def _models() -> list[tuple[str, str | None]]:
    return [(config.GEMINI_MODEL, config.GEMINI_THINKING_LEVEL), (config.GEMINI_FALLBACK_MODEL, None)]


def _config(model_thinking: str | None, **kwargs) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=model_thinking) if model_thinking else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **kwargs,
    )


def generate(prompt: str, *, system: str | None = None, json_schema: dict | None = None, temperature: float = 0.3) -> str:
    """Return Gemini's text. Tries the main model, then the fallback model, then raises GeminiUnavailable."""
    client = _client()
    errors = []
    for model, thinking in _models():
        cfg = _config(
            thinking,
            temperature=temperature,
            system_instruction=system,
            response_mime_type="application/json" if json_schema else None,
            response_json_schema=json_schema,
        )
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=cfg)
            if response.text:
                return response.text
            errors.append(f"{model}: empty reply")
        except Exception as exc:  # network, quota, bad model name: all mean "fall back"
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    raise GeminiUnavailable("; ".join(errors))


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def ping() -> tuple[bool, str]:
    """One tiny test call. Returns (ok, message)."""
    try:
        reply = generate("Reply with exactly the word OK.", temperature=0)
        return True, f"Gemini replied: {reply.strip()[:40]!r}"
    except Exception as exc:
        return False, f"Gemini test failed: {exc}"


# ================================================================= number check
# Every number Gemini writes (amounts, days, weeks, dates, invoice numbers,
# account references) must appear in the facts it was given. "£1,860" and
# "£1,860.00" count as the same number; "£1,900" does not.

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CENTS = Decimal("0.01")


def _decimal(value) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "")).copy_abs().quantize(CENTS)
    except (InvalidOperation, ValueError):
        return None


def numbers_in(text: str) -> list[tuple[str, Decimal]]:
    found = []
    for match in NUMBER_RE.finditer(text or ""):
        token = match.group().rstrip(",")
        value = _decimal(token)
        if value is not None:
            found.append((token, value))
    return found


def allowed_numbers(*facts) -> set[Decimal]:
    allowed: set[Decimal] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, bool) or x is None:
            return
        elif isinstance(x, (int, float, Decimal)):
            allowed.add(_decimal(x))
        else:
            allowed.update(v for _, v in numbers_in(str(x)))

    for f in facts:
        walk(f)
    return allowed


def unknown_numbers(text: str, allowed: set[Decimal]) -> list[str]:
    return sorted({token for token, value in numbers_in(text) if value not in allowed})


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("&", "and").split())


def check_text(text: str, allowed: set[Decimal], required_numbers=(), required_phrases=()) -> list[str]:
    """Return a list of problems (empty means the text passes)."""
    problems = []
    bad = unknown_numbers(text, allowed)
    if bad:
        problems.append("uses numbers that are not in the facts: " + ", ".join(bad))
    found = {value for _, value in numbers_in(text)}
    for n in required_numbers:
        if _decimal(n) not in found:
            problems.append(f"is missing the figure {money(n) if isinstance(n, float) else n}")
    for phrase in required_phrases:
        if phrase and _normalise(phrase) not in _normalise(text):
            problems.append(f"is missing '{phrase}'")
    return problems


# ======================================================================= drafts


@dataclass
class Draft:
    kind: str  # chase_overdue_invoices, query_bill or delay_payment
    label: str
    to: str
    subject: str
    body: str
    source: str  # "gemini" or "template"
    note: str = ""


DRAFT_SYSTEM = f"""You write short, plain-English business emails for {config.OWNER_NAME}, who owns {config.BUSINESS_NAME}, a small garage in Bradford.
Rules:
- Use only the facts given. Never invent, estimate or round any figure, date, reference, phone number or bank detail.
- Copy every figure exactly as it is written in the facts, for example £1,240.50, and write dates the way the facts do.
- Do not mention interest, legal action or late-payment law.
- Keep the body under 130 words. Sign off as: {config.OWNER_NAME}, {config.BUSINESS_NAME}.
- Return JSON with "subject" and "body"."""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"],
}


def write_draft(
    *,
    kind: str,
    label: str,
    to: str,
    brief: str,
    facts: dict,
    required_numbers: list,
    required_phrases: list[str],
    fallback: Callable[[], tuple[str, str]],
    use_gemini: bool = True,
) -> Draft:
    """Gemini drafts, Runway checks; one retry with the problems listed, then the template."""
    allowed = allowed_numbers(facts)
    note = "Gemini was not used"
    if use_gemini:
        problems: list[str] = []
        for attempt in range(2):
            prompt = f"Task: {brief}\n\nFacts (JSON):\n{json.dumps(facts, indent=1, ensure_ascii=False)}"
            if problems:
                prompt += "\n\nYour last draft was rejected because it " + "; it ".join(problems) + ". Write it again and fix this."
            try:
                data = parse_json(generate(prompt, system=DRAFT_SYSTEM, json_schema=DRAFT_SCHEMA, temperature=0.4))
            except GeminiUnavailable as exc:
                note = f"Gemini unavailable: {str(exc)[:120]}"
                break
            except (ValueError, json.JSONDecodeError) as exc:
                problems = [f"was not valid JSON ({exc})"]
                continue
            subject, body = str(data.get("subject", "")).strip(), str(data.get("body", "")).strip()
            problems = (["had an empty subject or body"] if not subject or not body else []) + check_text(
                f"{subject}\n{body}", allowed, required_numbers, required_phrases
            )
            if not problems:
                fixed = " after one fix" if attempt else ""
                return Draft(kind, label, to, subject, body, "gemini", f"Written by Gemini. Every figure checked against the data{fixed}.")
        else:
            note = "Gemini's draft failed the number check (" + "; ".join(problems) + ")"
    subject, body = fallback()
    return Draft(kind, label, to, subject, body, "template", f"Template used. {note}.")


TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def write_checked(
    *,
    task: str,
    facts: dict,
    system: str,
    required_numbers: list = (),
    required_phrases: list[str] = (),
    fallback: Callable[[], str],
    use_gemini: bool = True,
) -> tuple[str, str, str]:
    """Like write_draft, for a single piece of text (the spoken briefing, a spoken answer).

    Returns (text, source, note) where source is "gemini" or "template".
    """
    allowed = allowed_numbers(facts)
    note = "Gemini was not used"
    if use_gemini:
        problems: list[str] = []
        for attempt in range(2):
            prompt = f"Task: {task}\n\nFacts (JSON):\n{json.dumps(facts, indent=1, ensure_ascii=False)}"
            if problems:
                prompt += "\n\nYour last draft was rejected because it " + "; it ".join(problems) + ". Write it again and fix this."
            try:
                text = str(parse_json(generate(prompt, system=system, json_schema=TEXT_SCHEMA, temperature=0.4)).get("text", "")).strip()
            except GeminiUnavailable as exc:
                note = f"Gemini unavailable: {str(exc)[:120]}"
                break
            except (ValueError, json.JSONDecodeError) as exc:
                problems = [f"was not valid JSON ({exc})"]
                continue
            problems = (["was empty"] if not text else []) + check_text(text, allowed, required_numbers, required_phrases)
            if not problems:
                fixed = " after one fix" if attempt else ""
                return text, "gemini", f"Written by Gemini. Every figure checked against the data{fixed}."
        else:
            note = "Gemini's text failed the number check (" + "; ".join(problems) + ")"
    return fallback(), "template", f"Template used. {note}."


# ======================================================================== agent

ACTION_TYPES = ["chase_overdue_invoices", "query_bill", "delay_payment"]

TOOLS = {
    "get_cash_forecast": (
        "cash forecast",
        "Dave's weekly cash forecast for the next 8 weeks: cash today, the closing balance each week, "
        "the first week cash runs short, the payments due that week, and the effect of moving the one payment that can be moved.",
    ),
    "get_unpaid_invoices": (
        "unpaid invoices",
        "Customers who owe Dave money: overdue invoices (days overdue and reminder tone), invoices not yet due, "
        "totals, and the effect on the forecast if the overdue invoices are paid this week.",
    ),
    "get_supplier_bills": (
        "supplier bills",
        "Dave's monthly supplier bills (electricity, parts account, insurance, phone) for recent months, "
        "the usual level of each, and any month that spiked well above usual.",
    ),
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One or two plain-English sentences for Dave summing up the situation."},
        "actions": {
            "type": "array",
            "maxItems": 3,
            "description": "Up to 3 actions, most important first.",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {"type": "integer", "description": "1 = do this first."},
                    "type": {"type": "string", "enum": ACTION_TYPES},
                    "title": {"type": "string", "description": "Short title, at most 8 words."},
                    "why": {"type": "string", "description": "One or two sentences on why, quoting exact figures from the tools."},
                    "invoice_numbers": {"type": "array", "items": {"type": "string"}, "description": "For chase_overdue_invoices: which overdue invoices to chase."},
                    "supplier": {"type": "string", "description": "For query_bill or delay_payment: the supplier."},
                },
                "required": ["priority", "type", "title", "why"],
            },
        },
        "bill_suggestions": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string"},
            "description": "2 or 3 short, practical suggestions for dealing with the supplier bill spike, if there is one.",
        },
    },
    "required": ["headline", "actions", "bill_suggestions"],
}

AGENT_SYSTEM = f"""You are Runway, an AI financial health agent working for {config.OWNER_NAME}, who owns {config.BUSINESS_NAME}, a small garage in Bradford, Yorkshire.
{config.OWNER_NAME} is too busy to check the numbers. Your job is to look at his money, spot problems early, decide the few actions that will help most, and get the admin done for him.

How to work:
1. Call the tools to get the facts. You can call several tools at once. Every figure comes from these tools, which calculate it from {config.OWNER_NAME}'s own data.
2. Decide up to 3 actions in priority order (1 = do first). Protect cash before the week it runs short. Action types:
   - chase_overdue_invoices: Runway drafts a reminder for each overdue invoice you list, in the tone the tools give.
   - query_bill: Runway drafts a query email to a supplier whose bill spiked.
   - delay_payment: Runway drafts a polite request to move the movable payment to the later date the tools give.
3. Give 2 or 3 short, practical suggestions for dealing with any supplier bill spike.
4. Call submit_plan once with your decision. If Runway sends it back with problems, fix them and call submit_plan again.

Rules:
- Never invent, estimate or round a number. Copy figures exactly as the tools give them, for example £5,575.50.
- Plain English for a busy owner: short sentences, no jargon.
- Nothing is ever sent. {config.OWNER_NAME} reviews every draft before anything happens."""


def _agent_tools() -> list[types.Tool]:
    declarations = [types.FunctionDeclaration(name=name, description=desc) for name, (_, desc) in TOOLS.items()]
    declarations.append(
        types.FunctionDeclaration(
            name="submit_plan",
            description="Submit your decision: the headline, up to 3 prioritised actions, and suggestions for any bill spike.",
            parameters_json_schema=PLAN_SCHEMA,
        )
    )
    return [types.Tool(function_declarations=declarations)]


def validate_plan(raw: dict, context: dict) -> tuple[dict, list[str]]:
    """Check Gemini's plan against the data. Returns (clean plan, problems)."""
    problems: list[str] = []
    allowed = context["allowed"]
    headline = str(raw.get("headline") or "").strip()
    if not headline:
        problems.append("the headline is empty")

    def priority(a):
        try:
            return int(a.get("priority") or 99)
        except (TypeError, ValueError):
            return 99

    actions, seen = [], set()
    for a in sorted([a for a in (raw.get("actions") or []) if isinstance(a, dict)], key=priority):
        kind = a.get("type")
        if kind not in ACTION_TYPES:
            problems.append(f"{kind!r} is not an action type Runway can do")
            continue
        if kind in seen:
            continue
        if kind == "chase_overdue_invoices" and not context["overdue_numbers"]:
            problems.append("there are no overdue invoices to chase")
            continue
        if kind == "query_bill" and not context["spikes"]:
            problems.append("there is no bill spike to query")
            continue
        if kind == "delay_payment" and not context["movable"]:
            problems.append("there is no payment that can be moved")
            continue
        title, why = str(a.get("title") or "").strip(), str(a.get("why") or "").strip()
        if not title or not why:
            problems.append(f"the {kind} action needs a title and a why")
            continue
        action = {"priority": len(actions) + 1, "type": kind, "title": title, "why": why}
        if kind == "chase_overdue_invoices":
            chosen = [n for n in (a.get("invoice_numbers") or []) if n in context["overdue_numbers"]]
            action["invoice_numbers"] = chosen or list(context["overdue_numbers"])
        if kind == "query_bill":
            names = [s["supplier"] for s in context["spikes"]]
            action["supplier"] = a.get("supplier") if a.get("supplier") in names else names[0]
        if kind == "delay_payment":
            action["supplier"] = context["movable"]["supplier"]
        seen.add(kind)
        actions.append(action)
    actions = actions[:3]

    suggestions = [str(s).strip() for s in (raw.get("bill_suggestions") or []) if str(s).strip()][:3]
    if context["spikes"] and not suggestions:
        problems.append("give 2 or 3 suggestions for the bill spike")
    if context["runs_short"] and not actions:
        problems.append("give at least one action, because cash runs short")

    texts = [("the headline", headline)]
    texts += [(f"action {a['priority']}", f"{a['title']}\n{a['why']}") for a in actions]
    texts += [(f"suggestion {i + 1}", s) for i, s in enumerate(suggestions)]
    for where, text in texts:
        bad = unknown_numbers(text, allowed)
        if bad:
            problems.append(f"{where} uses numbers that are not in the data: {', '.join(bad)}")

    return {"headline": headline, "actions": actions, "bill_suggestions": suggestions}, problems


def _agent_loop(client, model, thinking, tool_results, context, say, deadline) -> dict:
    cfg = _config(
        thinking,
        temperature=0.2,
        system_instruction=AGENT_SYSTEM,
        tools=_agent_tools(),
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
    )
    task = (f"It is Monday {context['today']}. Check {config.BUSINESS_NAME}'s money and decide what "
            f"{config.OWNER_NAME} should do this week.")
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]
    for _ in range(config.AGENT_MAX_TURNS):
        if time.monotonic() > deadline:
            raise GeminiUnavailable("the agent ran out of time")
        response = client.models.generate_content(model=model, contents=contents, config=cfg)
        calls = response.function_calls or []
        model_turn = response.candidates[0].content if response.candidates else None
        if not calls:
            if model_turn:
                contents.append(model_turn)
            contents.append(types.Content(role="user", parts=[types.Part(text="Please call submit_plan now with your decision.")]))
            say("Gemini answered without using a tool, so Runway asked it to submit its plan")
            continue
        contents.append(model_turn)
        looked_up = [TOOLS[c.name][0] for c in calls if c.name in TOOLS]
        if looked_up:
            say("Gemini looked up: " + ", ".join(looked_up))
        replies, accepted = [], None
        for call in calls:
            if call.name in tool_results:
                result = {"result": tool_results[call.name]}
            elif call.name == "submit_plan":
                plan, problems = validate_plan(dict(call.args or {}), context)
                if problems:
                    result = {"status": "rejected", "problems": problems}
                    say("Runway sent the plan back to Gemini: " + "; ".join(problems))
                else:
                    result = {"status": "accepted"}
                    accepted = plan
            else:
                result = {"error": f"There is no tool called {call.name}"}
            replies.append(types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=result)))
        if accepted:
            say(f"Gemini decided {len(accepted['actions'])} actions. Every figure in the plan matches the data")
            return accepted
        contents.append(types.Content(role="user", parts=replies))
    raise GeminiUnavailable("Gemini did not submit a valid plan in time")


def run_agent(tool_results: dict, context: dict, say: Callable[[str], None], deadline: float) -> tuple[dict, str]:
    """Run the agent loop with the main model, then the fallback model. Returns (plan, model)."""
    client = _client()
    errors = []
    for model, thinking in _models():
        say(f"Asked Gemini ({model}) to review {config.BUSINESS_NAME}'s money")
        try:
            return _agent_loop(client, model, thinking, tool_results, context, say, deadline), model
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:160]}")
            say(f"{model} did not finish ({type(exc).__name__})")
            if time.monotonic() > deadline:
                break
    raise GeminiUnavailable("; ".join(errors))


def rules_plan(context: dict) -> dict:
    """Built-in plan used only when Gemini is unavailable and there is no saved analysis."""
    f, inv, bills_ = context["facts"]["get_cash_forecast"], context["facts"]["get_unpaid_invoices"], context["facts"]["get_supplier_bills"]
    actions = []
    if context["overdue_numbers"]:
        oldest = inv["overdue_invoices"][0]
        effect = inv.get("if_overdue_invoices_paid_this_week", {})
        tail = (f" If they pay this week, cash stays above zero for all {f['forecast_weeks']} weeks."
                if effect.get("cash_stays_above_zero_for_all_weeks") else "")
        actions.append({
            "type": "chase_overdue_invoices",
            "title": f"Chase {inv['overdue_count']} overdue invoices",
            "why": f"{inv['overdue_count']} customers owe {inv['overdue_total']}. The oldest is {oldest['days_overdue']} days overdue.{tail}",
            "invoice_numbers": list(context["overdue_numbers"]),
        })
    if context["spikes"]:
        s = bills_["spikes"][0]
        actions.append({
            "type": "query_bill",
            "title": f"Query the {s['month']} {s['bill'].lower()} bill",
            "why": f"{s['supplier']} charged {s['amount']} in {s['month']}, against a usual {s['usual_monthly_amount']} a month.",
            "supplier": s["supplier"],
        })
    if context["movable"]:
        m = f["movable_payment"]
        actions.append({
            "type": "delay_payment",
            "title": f"Ask to move the {m['due']} payment",
            "why": (f"Moving the {m['item']} payment of about {m['usual_amount']} from {m['due']} to {m['could_move_to']} "
                    f"leaves {m['balance_at_end_of_shortfall_week_if_moved']} at the end of week {f['runs_short_in_week']}."),
            "supplier": context["movable"]["supplier"],
        })
    for i, a in enumerate(actions[:3], 1):
        a["priority"] = i
    if f.get("runs_short_in_week"):
        headline = (f"Cash runs short in week {f['runs_short_in_week']} (week of {f['shortfall_week_starts']}), "
                    f"when it is forecast to reach {f['balance_at_end_of_shortfall_week']}.")
    else:
        headline = f"Cash stays above zero for the next {f['forecast_weeks']} weeks."
    import bills  # local import: bills imports this module

    suggestions = bills.fallback_suggestions(context["spikes"][0]) if context["spikes"] else []
    return {"headline": headline, "actions": actions[:3], "bill_suggestions": suggestions}


# ===================================================================== analyse


@dataclass
class AgentRun:
    plan: dict
    drafts: dict[str, list[Draft]]
    log: list[str]
    source: str  # "gemini", "cache" or "rules"
    seconds: float
    created_at: str
    model: str = ""
    notes: list[str] = field(default_factory=list)
    briefing: str = ""  # the spoken morning briefing (voice.py reads it aloud)
    briefing_source: str = ""  # "gemini" or "template"

    def to_json(self) -> dict:
        data = asdict(self)
        data["drafts"] = {k: [asdict(d) for d in v] for k, v in self.drafts.items()}
        return data

    @classmethod
    def from_json(cls, data: dict) -> "AgentRun":
        drafts = {k: [Draft(**d) for d in v] for k, v in data["drafts"].items()}
        return cls(**{**data, "drafts": drafts})


def data_fingerprint() -> str:
    h = hashlib.sha256()
    for path in (config.BANK_STATEMENT_CSV, config.INVOICES_CSV):
        h.update(path.read_bytes())
    h.update(f"{config.AS_OF_DATE}|{config.CACHE_VERSION}".encode())
    return h.hexdigest()


def save_cache(run: AgentRun) -> None:
    try:
        config.CACHE_FILE.parent.mkdir(exist_ok=True)
        config.CACHE_FILE.write_text(json.dumps({"fingerprint": data_fingerprint(), "run": run.to_json()}, indent=1, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        pass  # a cache is a nice-to-have


def load_cache() -> AgentRun | None:
    try:
        data = json.loads(config.CACHE_FILE.read_text(encoding="utf-8"))
        if data.get("fingerprint") != data_fingerprint():
            return None
        return AgentRun.from_json(data["run"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _draft_jobs(plan, statement, invoices, fc, use_gemini) -> list[tuple[str, Callable[[bool], Draft]]]:
    import bills
    import chaser

    jobs = []
    overdue = chaser.overdue_invoices(invoices, fc.as_of)
    spikes = bills.find_spikes(statement)
    for action in plan["actions"]:
        if action["type"] == "chase_overdue_invoices":
            for row in overdue[overdue["invoice_number"].isin(action["invoice_numbers"])].itertuples():
                jobs.append((action["type"], lambda g, row=row: chaser.draft_reminder(row, use_gemini=g)))
        elif action["type"] == "query_bill":
            spike = next((s for s in spikes if s["supplier"] == action.get("supplier")), spikes[0])
            jobs.append((action["type"], lambda g, spike=spike: bills.draft_query_email(spike, use_gemini=g)))
        elif action["type"] == "delay_payment" and fc.delay_candidate:
            jobs.append((action["type"], lambda g: bills.draft_delay_email(fc, statement, use_gemini=g)))
    return jobs


def make_drafts(plan, statement, invoices, fc, use_gemini: bool, say: Callable[[str], None]) -> dict[str, list[Draft]]:
    """Draft every email the plan needs, in parallel. Anything not back in time gets its template."""
    jobs = _draft_jobs(plan, statement, invoices, fc, use_gemini)
    results: list[Draft | None] = [None] * len(jobs)
    if use_gemini and jobs:
        say(f"Gemini is drafting {len(jobs)} emails")
        pool = ThreadPoolExecutor(max_workers=min(6, len(jobs)))
        futures = {pool.submit(job, True): i for i, (_, job) in enumerate(jobs)}
        done, _ = wait(futures, timeout=config.DRAFTS_DEADLINE_SECONDS)
        for future in done:
            try:
                results[futures[future]] = future.result()
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)
    drafts: dict[str, list[Draft]] = {}
    for i, (kind, job) in enumerate(jobs):
        draft = results[i] or job(False)
        if results[i] is None and use_gemini:
            draft.note = "Template used. Gemini did not finish this draft in time."
        drafts.setdefault(kind, []).append(draft)
        say(f"Draft for {draft.label}: " + ("written by Gemini, figures checked ✓" if draft.source == "gemini" else "template"))
    return drafts


def agent_facts(statement, invoices, fc: Forecast) -> dict:
    """The facts behind every tool, calculated in code. Also used to answer Dave's questions."""
    import bills
    import chaser

    return {
        "get_cash_forecast": forecast_facts(fc),
        "get_unpaid_invoices": chaser.invoice_facts(invoices, fc),
        "get_supplier_bills": bills.bill_facts(statement),
    }


def analyse(
    statement, invoices, fc: Forecast, progress: Callable[[str], None] | None = None,
    use_cache: bool = True, allow_gemini: bool = True,
) -> AgentRun:
    """Run Runway's agent end to end: facts → Gemini plan → checked drafts and briefing. Never raises for API problems."""
    import bills
    import chaser
    import voice

    started = time.monotonic()
    log: list[str] = []

    def say(message: str) -> None:
        log.append(message)
        if progress:
            progress(message)

    say("Runway calculated the forecast, the unpaid invoices and the supplier bills from the data")
    tool_results = agent_facts(statement, invoices, fc)
    spikes = bills.find_spikes(statement)
    movable = None
    if fc.delay_candidate:
        movable = {"supplier": config.SUPPLIERS[fc.delay_candidate["category"]]["name"]}
    context = {
        "today": short_date(fc.as_of),
        "facts": tool_results,
        "allowed": allowed_numbers(tool_results),
        "overdue_numbers": chaser.overdue_invoices(invoices, fc.as_of)["invoice_number"].tolist(),
        "spikes": spikes,
        "movable": movable,
        "runs_short": fc.runs_short,
    }

    model, source = "", "gemini"
    try:
        if not allow_gemini:
            raise GeminiUnavailable("switched off for a rehearsal")
        if not config.gemini_api_key():
            raise GeminiUnavailable("GEMINI_API_KEY is not set")
        plan, model = run_agent(tool_results, context, say, started + config.AGENT_DEADLINE_SECONDS)
    except GeminiUnavailable as exc:
        say(f"Gemini is unavailable ({str(exc)[:120]})")
        cached = load_cache() if use_cache else None
        if cached:
            say(f"Showing Runway's last verified analysis, saved {cached.created_at}")
            cached.log = log + ["(Saved run) " + line for line in cached.log]
            cached.source, cached.seconds = "cache", round(time.monotonic() - started, 1)
            if not cached.briefing:
                cached.briefing, cached.briefing_source, _ = voice.write_briefing(fc, tool_results, cached.plan, use_gemini=False)
            return cached
        say("Showing Runway's built-in plan. Every figure is still calculated from the data")
        plan, source = rules_plan(context), "rules"

    use_gemini = source == "gemini"
    # The spoken briefing is written alongside the drafts, so it is ready when Dave presses play.
    briefing_pool = ThreadPoolExecutor(max_workers=1)
    briefing_job = briefing_pool.submit(voice.write_briefing, fc, tool_results, plan, use_gemini)
    drafts = make_drafts(plan, statement, invoices, fc, use_gemini=use_gemini, say=say)
    try:
        briefing, briefing_source, _ = briefing_job.result(timeout=config.DRAFTS_DEADLINE_SECONDS)
    except Exception:
        briefing, briefing_source, _ = voice.write_briefing(fc, tool_results, plan, use_gemini=False)
    briefing_pool.shutdown(wait=False, cancel_futures=True)
    say("Morning briefing: " + ("written by Gemini, figures checked ✓" if briefing_source == "gemini" else "template"))

    seconds = round(time.monotonic() - started, 1)
    say(f"Finished in {seconds} seconds. Nothing has been sent")
    run = AgentRun(plan=plan, drafts=drafts, log=log, source=source, seconds=seconds,
                   created_at=datetime.now().strftime("%d %b %Y, %H:%M"), model=model,
                   briefing=briefing, briefing_source=briefing_source)
    if source == "gemini":
        save_cache(run)
    return run


if __name__ == "__main__":
    ok, message = ping()
    print(("PASS " if ok else "FAIL ") + message)
