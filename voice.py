"""ElevenLabs voice for Runway: the morning briefing and spoken questions.

  * Briefing: Gemini writes a 30-second script from figures calculated in code
    (rounded to whole pounds for speech, also in code). The number check applies.
    ElevenLabs reads it aloud.
  * Questions: ElevenLabs turns Dave's recording into text, Gemini answers from the
    data (number check applies; "who owes me money?" must name every customer who
    owes money and the right amounts), and ElevenLabs speaks the answer.
  * Fallbacks: if ElevenLabs fails, Runway plays the last recording of the same text
    saved on this machine, or shows the text. If Gemini fails, a template written
    from the same figures is used. The demo never stops.

Run `python voice.py` to make one short test call with your ELEVENLABS_API_KEY.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from elevenlabs.client import ElevenLabs

import config
import llm
from forecast import Forecast


class VoiceUnavailable(Exception):
    """No key, or the ElevenLabs call failed. Callers show the text instead."""


# ====================================================================== client


def _client() -> ElevenLabs:
    key = config.elevenlabs_api_key()
    if not key:
        raise VoiceUnavailable("ELEVENLABS_API_KEY is not set")
    return ElevenLabs(api_key=key, timeout=config.ELEVENLABS_TIMEOUT_SECONDS)


def text_to_speech(text: str) -> bytes:
    """Return MP3 bytes for the text, or raise VoiceUnavailable."""
    try:
        chunks = _client().text_to_speech.convert(
            voice_id=config.ELEVENLABS_VOICE_ID,
            text=text,
            model_id=config.ELEVENLABS_TTS_MODEL,
            output_format=config.ELEVENLABS_OUTPUT_FORMAT,
        )
        audio = b"".join(chunks)
    except VoiceUnavailable:
        raise
    except Exception as exc:
        raise VoiceUnavailable(f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    if not audio:
        raise VoiceUnavailable("ElevenLabs returned no audio")
    return audio


def transcribe(audio: bytes, filename: str = "question.wav", content_type: str = "audio/wav", allow: bool = True) -> str:
    """Turn Dave's recorded question into text with ElevenLabs speech-to-text."""
    if not allow:
        raise VoiceUnavailable("switched off for a rehearsal")
    try:
        result = _client().speech_to_text.convert(
            file=(filename, audio, content_type),
            model_id=config.ELEVENLABS_STT_MODEL,
            language_code="eng",
            tag_audio_events=False,
        )
    except VoiceUnavailable:
        raise
    except Exception as exc:
        raise VoiceUnavailable(f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    text = (getattr(result, "text", "") or "").strip()
    if not text:
        raise VoiceUnavailable("No speech was recognised")
    return text


def ping() -> tuple[bool, str]:
    """One short test call (a few characters of speech). Returns (ok, message)."""
    try:
        audio = text_to_speech("Runway is ready.")
        return True, f"ElevenLabs returned {len(audio):,} bytes of audio"
    except Exception as exc:
        return False, f"ElevenLabs test failed: {exc}"


# ============================================================== speaking aloud

MONEY_RE = re.compile(r"(-)?£(\d[\d,]*)(?:\.(\d{2}))?")


def speakable(text: str) -> str:
    """Write amounts the way they are said: '£1,240.50' becomes '1,240 pounds 50'."""

    def say(match: re.Match) -> str:
        minus, pounds, pence = match.groups()
        spoken = f"{'minus ' if minus else ''}{pounds} pounds"
        return spoken + (f" {pence}" if pence and pence != "00" else "")

    return MONEY_RE.sub(say, text)


def _audio_path(spoken: str):
    key = hashlib.sha256(f"{spoken}|{config.ELEVENLABS_VOICE_ID}|{config.ELEVENLABS_TTS_MODEL}".encode()).hexdigest()[:24]
    return config.CACHE_FILE.parent / "audio" / f"{key}.mp3"


def speak(text: str, allow: bool = True) -> tuple[bytes, str]:
    """Return (mp3 bytes, "live" or "cached"). Saves each recording so it can be replayed if ElevenLabs fails later."""
    spoken = speakable(text)
    path = _audio_path(spoken)
    try:
        if not allow:
            raise VoiceUnavailable("switched off for a rehearsal")
        audio = text_to_speech(spoken)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        except OSError:
            pass
        return audio, "live"
    except VoiceUnavailable:
        if path.exists():
            return path.read_bytes(), "cached"
        raise


# =================================================================== briefing


def spoken_money(value: float) -> str:
    """Whole pounds for speech, rounded in code: 1952.24 -> '£1,952'."""
    pounds = Decimal(str(abs(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"£{pounds:,}"


SHORT_NAMES = {"vat": "VAT", "parts_account": "the parts account", "rent": "rent", "electricity": "electricity",
               "equipment_finance": "equipment finance", "insurance": "insurance", "phone": "the phone bill"}


def briefing_facts(fc: Forecast, facts: dict, plan: dict) -> dict:
    inv, b = facts["get_unpaid_invoices"], facts["get_supplier_bills"]
    d = fc.as_of
    out = {
        "owner": config.OWNER_NAME,
        "business": config.BUSINESS_NAME,
        "day": f"{d:%A} {d.day} {d:%B}",
        "cash_today": spoken_money(fc.current_balance),
        "weeks_of_cash_left": fc.weeks_of_cash_left,
    }
    if fc.runs_short:
        start = fc.shortfall_week_start
        week = fc.flows[(fc.flows["week"] == fc.shortfall_week) & (fc.flows["amount"] < 0) & (fc.flows["category"].isin(SHORT_NAMES))]
        out.update(
            runs_short_in_week=fc.shortfall_week,
            week_starting=f"{start.day} {start:%B}",
            short_by=spoken_money(fc.shortfall_balance),
            main_payments_that_week=[f"{SHORT_NAMES[r.category]} {spoken_money(r.amount)}" for r in week.sort_values("amount").head(2).itertuples()],
        )
    if inv["overdue_count"]:
        overdue_total = sum(float(i["amount"].replace("£", "").replace(",", "")) for i in inv["overdue_invoices"])
        out["overdue_invoices"] = inv["overdue_count"]
        out["overdue_total"] = spoken_money(overdue_total)
        effect = inv.get("if_overdue_invoices_paid_this_week", {})
        if effect.get("cash_stays_above_zero_for_all_weeks"):
            out["if_paid_this_week"] = "cash stays above zero for the whole forecast"
    if b["spikes"]:
        s = b["spikes"][0]
        amount = float(s["amount"].replace("£", "").replace(",", ""))
        usual = float(s["usual_monthly_amount"].replace("£", "").replace(",", ""))
        month = datetime.strptime(s["month"], "%b %Y").strftime("%B")  # "Sep 2026" -> "September"
        out["bill_spike"] = {"supplier": s["supplier"], "bill": s["bill"].lower(), "month": month,
                             "amount": spoken_money(amount), "usual": spoken_money(usual)}
    out["actions_decided"] = [a["title"] for a in plan["actions"]]
    return out


BRIEFING_SYSTEM = f"""You write the spoken morning briefing for {config.OWNER_NAME}, who owns {config.BUSINESS_NAME}. A voice reads it aloud, so:
- 60 to 80 words. Short spoken sentences. No lists, symbols, emojis or headings.
- Use only the facts given. Copy every figure exactly as written, as digits (for example £1,952 and week 5). Never invent or change a number.
- Start with "Good morning {config.OWNER_NAME}." Cover cash and when it runs short, the overdue invoices, the bill spike and what Runway recommends.
- End by saying the draft emails are ready to check and nothing has been sent.
Return JSON with "text"."""


def briefing_template(facts: dict) -> str:
    parts = [f"Good morning {facts['owner']}. You have {facts['cash_today']} in the bank, about {facts['weeks_of_cash_left']} weeks of cash."]
    if "runs_short_in_week" in facts:
        due = " and ".join(p.split(" £")[0] for p in facts["main_payments_that_week"])
        parts.append(f"Cash runs short in week {facts['runs_short_in_week']}, the week of {facts['week_starting']}, "
                     f"when {due} fall due. You'd be {facts['short_by']} short.")
    if "overdue_total" in facts:
        tail = ", and if they pay this week, cash stays above zero" if "if_paid_this_week" in facts else ""
        parts.append(f"{facts['overdue_invoices']} customers owe you {facts['overdue_total']} in overdue invoices{tail}.")
    if "bill_spike" in facts:
        s = facts["bill_spike"]
        parts.append(f"Your {s['month']} {s['bill']} bill was {s['amount']}, against a usual {s['usual']}, so it's worth querying.")
    parts.append("Your draft emails are ready to check. Nothing has been sent.")
    return " ".join(parts)


def write_briefing(fc: Forecast, facts: dict, plan: dict, use_gemini: bool = True) -> tuple[str, str, str]:
    """Return (briefing text, "gemini" or "template", note)."""
    bf = briefing_facts(fc, facts, plan)
    required = []
    if "runs_short_in_week" in bf:
        required = [bf["runs_short_in_week"], bf["short_by"].replace("£", "")]
    return llm.write_checked(
        task="Write this morning's spoken briefing.",
        facts=bf,
        system=BRIEFING_SYSTEM,
        required_numbers=required,
        fallback=lambda: briefing_template(bf),
        use_gemini=use_gemini,
    )


# ================================================================== questions

OWES_RE = re.compile(r"\bow(e|es|ed|ing)\b|\bdebt|\bunpaid|\boverdue|\binvoice|\bchase|\bpaid me\b", re.IGNORECASE)
CASH_RE = re.compile(r"\bcash|\bmoney left|\brunway|\bshort|\bbank|\bbalance|\bweeks?\b|\bforecast", re.IGNORECASE)
BILL_RE = re.compile(r"\bbill|\belectric|\benergy|\bsupplier|\bspike", re.IGNORECASE)

ANSWER_SYSTEM = f"""You are Runway, answering {config.OWNER_NAME}'s spoken question about the money in his business, {config.BUSINESS_NAME}.
- Answer in plain spoken English: 1 to 4 short sentences, under 80 words. No lists, symbols or headings.
- Use only the facts given. Copy every figure exactly as written (for example £2,475.00). Never invent or change a number.
- If he asks who owes him money, name every customer in money_owed with their amount, saying which are overdue and by how many days.
- If the facts do not cover the question, say so in one sentence and say what you can help with: cash, who owes him money, bills and this week's actions.
Return JSON with "text"."""


def answer_facts(question: str, facts: dict, plan: dict) -> dict:
    return {
        "question": question,
        "forecast": facts["get_cash_forecast"],
        "money_owed": facts["get_unpaid_invoices"],
        "supplier_bills": facts["get_supplier_bills"],
        "actions_this_week": [a["title"] for a in plan["actions"]],
    }


def owes_answer(inv: dict) -> str:
    """Template answer to "who owes me money?"."""
    overdue, upcoming = inv["overdue_invoices"], inv["not_yet_due"]
    if not overdue and not upcoming:
        return "Nobody owes you money right now."
    text = f"{inv['customers_who_owe_money']} customers owe you {inv['total_owed']}."
    if overdue:
        text += " Overdue: " + "; ".join(f"{i['customer']}, {i['amount']}, {i['days_overdue']} days late" for i in overdue) + "."
    if upcoming:
        text += " Not due yet: " + "; ".join(f"{i['customer']}, {i['amount']}, due {i['due_date']}" for i in upcoming) + "."
    return text


def template_answer(question: str, facts: dict) -> str:
    f, inv, b = facts["get_cash_forecast"], facts["get_unpaid_invoices"], facts["get_supplier_bills"]
    if OWES_RE.search(question):
        return owes_answer(inv)
    if BILL_RE.search(question) and b["spikes"]:
        s = b["spikes"][0]
        return (f"Your {s['month']} {s['bill'].lower()} bill from {s['supplier']} was {s['amount']}, against a usual "
                f"{s['usual_monthly_amount']} a month. The query email is drafted and ready to check.")
    if CASH_RE.search(question):
        if f.get("runs_short_in_week"):
            return (f"You have {f['cash_today']} today. Cash runs short in week {f['runs_short_in_week']}, the week of "
                    f"{f['shortfall_week_starts']}, when the balance is forecast to reach {f['balance_at_end_of_shortfall_week']}.")
        return f"You have {f['cash_today']} today, and cash stays above zero for the next {f['forecast_weeks']} weeks."
    return "I can tell you who owes you money, when cash runs short, or about your bills. Try asking: who owes me money?"


def answer_question(question: str, facts: dict, plan: dict, use_gemini: bool = True) -> tuple[str, str, str]:
    """Return (answer text, "gemini" or "template", note). Every figure comes from the data."""
    af = answer_facts(question, facts, plan)
    required_numbers, required_phrases = [], []
    if OWES_RE.search(question):  # must name every customer who owes money, with the right amounts
        inv = facts["get_unpaid_invoices"]
        rows = inv["overdue_invoices"] + inv["not_yet_due"]
        required_phrases = list(dict.fromkeys(r["customer"] for r in rows))
        required_numbers = [r["amount"].replace("£", "") for r in rows]
    return llm.write_checked(
        task=f"Answer this question: {question}",
        facts=af,
        system=ANSWER_SYSTEM,
        required_numbers=required_numbers,
        required_phrases=required_phrases,
        fallback=lambda: template_answer(question, facts),
        use_gemini=use_gemini,
    )


if __name__ == "__main__":
    ok, message = ping()
    print(("PASS " if ok else "FAIL ") + message)
