"""Automated checks for the test pack (TC1 to TC3 here; TC4 is added in Step 3).

These run offline against a simulated Gemini (tests/fake_gemini.py), so they prove
the calculations, the agent loop, the number check and every fallback. What the
real Gemini writes is checked by tests/test_live.py and by hand.

Run:  pytest -q
"""

import re
from pathlib import Path

import bills
import chaser
import config
import llm
from forecast import money

ROOT = Path(__file__).resolve().parents[1]
OVERDUE = {"INV-1029": ("Shipley Van Hire", 2475.00, 49, "final"),
           "INV-1037": ("Calder Plumbing & Heating", 1240.50, 31, "firm"),
           "INV-1042": ("Wharfe Valley Taxis", 1860.00, 12, "friendly")}


def all_numbers_in_data(run: llm.AgentRun, statement, invoices, fc) -> bool:
    facts = {"f": llm.forecast_facts(fc), "i": chaser.invoice_facts(invoices, fc), "b": bills.bill_facts(statement)}
    text = run.plan["headline"] + " ".join(a["title"] + a["why"] for a in run.plan["actions"]) + " ".join(run.plan["bill_suggestions"])
    return llm.unknown_numbers(text, llm.allowed_numbers(facts)) == []


# =============================================================== TC1 forecast


def test_tc1_forecast_says_cash_runs_short_in_week_5(data):
    _, _, fc = data
    assert fc.warning == "Cash runs short in week 5"
    assert fc.shortfall_week == 5 and fc.weeks_of_cash_left == 4


def test_tc1_weekly_chart_dips_below_zero_in_week_5(data):
    _, _, fc = data
    closing = dict(zip(fc.weeks["week"], fc.weeks["closing_balance"]))
    assert all(closing[w] >= 0 for w in (1, 2, 3, 4))
    assert closing[5] < 0


def test_tc1_gemini_agent_lists_exactly_three_actions(fake_gemini, data):
    fake = fake_gemini()
    run = llm.analyse(*data)
    assert run.source == "gemini"
    assert [a["type"] for a in run.plan["actions"]] == ["chase_overdue_invoices", "query_bill", "delay_payment"]
    assert all_numbers_in_data(run, *data)
    tool_turn = [c for _, contents, cfg in fake.requests if cfg.tools for c in contents]
    assert tool_turn, "the agent should have used its tools"


def test_tc1_invented_number_in_plan_is_sent_back_and_fixed(fake_gemini, data):
    fake_gemini(plan_mistake=True)
    run = llm.analyse(*data)
    assert run.source == "gemini"
    assert any("sent the plan back" in line and "5,600" in line for line in run.log)
    assert "5,600" not in run.plan["headline"]


def test_tc1_gemini_down_still_shows_forecast_and_plain_actions(fake_gemini, data):
    fake_gemini(fail=True)
    run = llm.analyse(*data)
    assert run.source == "rules"
    assert len(run.plan["actions"]) == 3
    assert all_numbers_in_data(run, *data)


def test_tc1_last_verified_run_is_reused_if_gemini_fails_later(fake_gemini, data):
    fake_gemini()
    first = llm.analyse(*data)
    assert config.CACHE_FILE.exists()
    fake_gemini(fail=True)
    second = llm.analyse(*data)
    assert second.source == "cache"
    assert second.plan == first.plan


def test_tc1_no_key_uses_built_in_plan(no_keys, data):
    run = llm.analyse(*data, use_cache=False)
    assert run.source == "rules" and len(run.plan["actions"]) == 3


# ============================================================ TC2 chaser


def test_tc2_exactly_three_overdue_invoices_found(data):
    _, invoices, fc = data
    overdue = chaser.overdue_invoices(invoices, fc.as_of)
    found = {r.invoice_number: (r.customer, r.amount, r.days_overdue, r.tone) for r in overdue.itertuples()}
    assert found == OVERDUE


def check_reminders(drafts):
    assert len(drafts) == 3
    for d in drafts:
        number = next(n for n in OVERDUE if n in d.subject + d.body)
        customer, amount, days, _ = OVERDUE[number]
        text = d.to + d.subject + d.body
        assert customer in d.to and customer.split()[0] in d.subject + d.body
        assert money(amount) in text and f"{days} days" in text
        others = [money(a) for n, (_, a, _, _) in OVERDUE.items() if n != number]
        assert not any(o in d.subject + d.body for o in others), "a draft must not quote another customer's amount"


def test_tc2_each_gemini_draft_names_right_customer_amount_and_days(fake_gemini, data):
    fake_gemini()
    run = llm.analyse(*data)
    drafts = run.drafts["chase_overdue_invoices"]
    assert all(d.source == "gemini" for d in drafts)
    check_reminders(drafts)


def test_tc2_invented_number_in_draft_is_caught_and_fixed(fake_gemini, data):
    fake_gemini(draft_mistake_for={"Shipley Van Hire"})
    run = llm.analyse(*data)
    shipley = next(d for d in run.drafts["chase_overdue_invoices"] if "Shipley" in d.to)
    assert shipley.source == "gemini" and "after one fix" in shipley.note
    assert "01274" not in shipley.body


def test_tc2_draft_that_keeps_failing_falls_back_to_template(fake_gemini, data):
    fake_gemini(always_bad_drafts=True)
    run = llm.analyse(*data)
    drafts = run.drafts["chase_overdue_invoices"]
    assert all(d.source == "template" for d in drafts)
    check_reminders(drafts)


def test_tc2_template_reminders_are_right_without_gemini(no_keys, data):
    run = llm.analyse(*data, use_cache=False)
    check_reminders(run.drafts["chase_overdue_invoices"])


def test_tc2_reminder_tone_matches_how_overdue(data):
    assert [chaser.tone_for(d) for d in (12, 31, 49)] == ["friendly", "firm", "final"]


def test_tc2_nothing_is_ever_sent():
    source = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.glob("*.py"))
    for forbidden in ("smtplib", "sendmail", "send_message", "gmail", "sendgrid", "mailgun", "requests.post"):
        assert forbidden not in source, f"found {forbidden!r}: Runway must never send email"
    assert not hasattr(llm.Draft, "send")


# ======================================================== TC3 bill checker


def test_tc3_exactly_one_spike_and_it_is_september_electricity(data):
    statement, _, _ = data
    monthly = bills.monthly_supplier_costs(statement)
    spikes = monthly[monthly["is_spike"]]
    assert len(spikes) == 1
    assert (spikes["bill"].iloc[0], spikes["month"].iloc[0]) == ("Electricity", "Sep 2026")
    electricity = monthly[monthly["bill"] == "Electricity"]
    assert electricity["is_spike"].sum() == 1 and len(electricity) >= 6  # several months, one highlighted


def test_tc3_usual_level_and_spike_figures(data):
    statement, _, _ = data
    (spike,) = bills.find_spikes(statement)
    assert (spike["amount"], spike["usual"], spike["excess"], spike["account_ref"]) == (1236.80, 391.30, 845.50, "88213")


def test_tc3_gemini_suggests_actions_for_the_spike(fake_gemini, data):
    fake_gemini()
    run = llm.analyse(*data)
    suggestions = run.plan["bill_suggestions"]
    assert 2 <= len(suggestions) <= 3
    assert all_numbers_in_data(run, *data)


def test_tc3_query_email_quotes_the_real_amounts(fake_gemini, data):
    fake_gemini()
    run = llm.analyse(*data)
    (draft,) = run.drafts["query_bill"]
    text = draft.subject + draft.body
    assert "£1,236.80" in text and "£391.30" in text and "88213" in text
    assert "Northern Grid Energy" in draft.to


# ======================================================= number check (NFR2)


def test_number_check_rejects_invented_or_rounded_figures():
    allowed = llm.allowed_numbers({"amount": "£1,860.00", "days": 12, "invoice": "INV-1042"})
    assert llm.unknown_numbers("Invoice INV-1042 for £1,860 is 12 days late", allowed) == []
    assert llm.unknown_numbers("about £1,900, 14 days late", allowed) == ["1,900", "14"]
    problems = llm.check_text("Invoice INV-1042 is late", allowed, required_numbers=[1860.00, 12])
    assert len(problems) == 2


def test_api_keys_are_never_in_the_code():
    key_like = re.compile(r"\bsk_[0-9a-f]{20,}|AIza[0-9A-Za-z_-]{20,}|\bAQ\.[0-9A-Za-z_-]{20,}")
    for path in list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + list(ROOT.glob("*.csv")):
        assert not key_like.search(path.read_text(encoding="utf-8")), f"something that looks like an API key is in {path.name}"
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()


# ================================================================ TC4 voice

import pytest  # noqa: E402
from fake_elevenlabs import FakeElevenLabs  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import voice  # noqa: E402

UNPAID = {"Shipley Van Hire": "£2,475.00", "Calder Plumbing & Heating": "£1,240.50", "Wharfe Valley Taxis": "£1,860.00",
          "Bingley Florists": "£385.00", "Airedale Couriers": "£1,120.00"}


@pytest.fixture
def fake_voice(monkeypatch):
    def install(**options) -> FakeElevenLabs:
        fake = FakeElevenLabs(**options)
        monkeypatch.setattr(voice, "_client", lambda: fake)
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
        return fake

    return install


def names_everyone_who_owes(text: str) -> bool:
    return all(customer in text and amount in text for customer, amount in UNPAID.items())


def test_tc4_gemini_briefing_matches_the_figures(fake_gemini, data):
    fake_gemini()
    run = llm.analyse(*data)
    assert run.briefing_source == "gemini"
    for figure in ("£3,600", "week 5", "£1,952", "£5,576", "Nothing has been sent"):
        assert figure in run.briefing


def test_tc4_template_briefing_matches_the_figures_and_lasts_about_30_seconds(no_keys, data):
    run = llm.analyse(*data, use_cache=False)
    for figure in ("£3,600", "4 weeks", "week 5", "2 November", "£1,952", "£5,576", "September", "£1,237", "£391"):
        assert figure in run.briefing
    assert 60 <= len(run.briefing.split()) <= 90  # about 30 seconds at a normal speaking pace


def test_tc4_briefing_is_read_by_elevenlabs_with_amounts_said_aloud(fake_voice, no_keys, data):
    run = llm.analyse(*data, use_cache=False)
    fake = fake_voice()
    audio, how = voice.speak(run.briefing)
    assert how == "live" and audio.startswith(b"ID3")
    assert "3,600 pounds" in fake.spoken[0] and "£" not in fake.spoken[0]


def test_tc4_spoken_question_who_owes_me_money(fake_gemini, fake_voice, data):
    fake_gemini()
    fake = fake_voice(transcript="Who owes me money?")
    run = llm.analyse(*data)
    question = voice.transcribe(b"RIFF....WAVE")
    text, source, _ = voice.answer_question(question, llm.agent_facts(*data), run.plan)
    assert question == "Who owes me money?" and source == "gemini"
    assert names_everyone_who_owes(text)
    voice.speak(text)
    assert "2,475 pounds" in fake.spoken[-1]


def test_tc4_answer_that_leaves_a_customer_out_is_caught(fake_gemini, data):
    fake_gemini(skip_customer="Bingley Florists")
    run = llm.analyse(*data)
    text, source, note = voice.answer_question("who owes me money?", llm.agent_facts(*data), run.plan)
    assert source == "template" and "Bingley Florists" in note
    assert names_everyone_who_owes(text)


def test_tc4_template_answer_names_the_right_customers_and_amounts(no_keys, data):
    run = llm.analyse(*data, use_cache=False)
    text, source, _ = voice.answer_question("Who owes me money?", llm.agent_facts(*data), run.plan, use_gemini=False)
    assert source == "template" and names_everyone_who_owes(text)


def test_tc4_without_elevenlabs_key_voice_says_unavailable(no_keys):
    with pytest.raises(voice.VoiceUnavailable):
        voice.speak("Good morning Dave.")
    with pytest.raises(voice.VoiceUnavailable):
        voice.transcribe(b"RIFF....WAVE")


def test_tc4_saved_recording_is_replayed_if_elevenlabs_fails_later(fake_voice):
    fake_voice()
    first, how = voice.speak("Good morning Dave. You have £3,600.")
    assert how == "live"
    fake_voice(fail=True)
    again, how = voice.speak("Good morning Dave. You have £3,600.")
    assert how == "cached" and again == first


# ========================================================= whole screen (UI)


def load_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(b for b in at.button if b.label.startswith("Load")).click().run()
    assert not at.exception, at.exception
    return at


def click(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception
    return at


def test_app_full_demo_flow_on_one_screen(fake_gemini, fake_voice):
    fake_gemini()
    fake_voice()
    at = load_app()
    assert "SAMPLE DATA" in at.warning[0].value
    assert any("Cash runs short in week 5" in e.value for e in at.error)
    assert sum(b.label.startswith("Open") for b in at.button) >= 3
    click(at, "Open 3 draft reminders")
    assert len(at.code) >= 4  # bank export + 3 reminders
    click(at, "Play the 30-second briefing")
    assert len(at.get("audio")) == 1
    click(at, "Who owes me money?")
    assert names_everyone_who_owes(at.success[-1].value)


def test_app_without_elevenlabs_key_shows_text_briefing(fake_gemini, monkeypatch, no_env_file):
    fake_gemini()
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    at = load_app()
    click(at, "Play the 30-second briefing")
    assert len(at.get("audio")) == 0
    assert any("here is the briefing as text" in i.value for i in at.info)
    assert any("week 5" in m.value for m in at.markdown)
    at.text_input[0].input("who owes me money?").run()
    click(at, "Ask")
    assert names_everyone_who_owes(at.success[-1].value)


def test_app_with_gemini_down_still_shows_forecast_and_actions(fake_gemini, fake_voice):
    fake_gemini(fail=True)
    fake_voice()
    at = load_app()
    assert any("Cash runs short in week 5" in e.value for e in at.error)
    assert any("built-in plan" in w.value for w in at.warning)
    assert sum(b.label.startswith("Open") for b in at.button) >= 3


def test_keys_are_read_fresh_from_env_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert config.gemini_api_key() is None
    env.write_text("\ufeffGEMINI_API_KEY = 'abc123'\n", encoding="utf-8")  # BOM and quotes, as Notepad might save it
    assert config.gemini_api_key() == "abc123"
    env.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    assert config.gemini_api_key() is None
