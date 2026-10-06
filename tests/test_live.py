"""Live checks against the real Gemini and ElevenLabs APIs. Each is skipped if its key is not set.

Run on the demo machine and network, just before presenting:
    pytest -q tests/test_live.py -s

A pass also saves the fallbacks the app uses if an API fails mid-demo:
cache/last_agent_run.json (Gemini's last verified analysis) and the recording of
its briefing in cache/audio/.
"""

import time

import pytest

import config
import llm
import voice
from test_runway import check_reminders, names_everyone_who_owes

needs_gemini = pytest.mark.skipif(not config.gemini_api_key(), reason="GEMINI_API_KEY is not set")
needs_voice = pytest.mark.skipif(not config.elevenlabs_api_key(), reason="ELEVENLABS_API_KEY is not set")


@pytest.fixture(autouse=True)
def real_cache(monkeypatch):
    """Undo the throwaway cache from conftest, so good live runs are saved for the demo."""
    monkeypatch.setattr(config, "CACHE_FILE", config.BASE_DIR / "cache" / "last_agent_run.json")


@needs_gemini
def test_live_gemini_connection():
    ok, message = llm.ping()
    assert ok, message


@needs_gemini
def test_live_agent_tc1_to_tc3(data):
    run = llm.analyse(*data, use_cache=False)
    print("\n".join(run.log))
    assert run.source == "gemini", "Gemini did not finish; see the log above"
    assert len(run.plan["actions"]) == 3, f"Gemini chose {len(run.plan['actions'])} actions"
    assert {a["type"] for a in run.plan["actions"]} == set(llm.ACTION_TYPES)
    assert 2 <= len(run.plan["bill_suggestions"]) <= 3
    check_reminders(run.drafts["chase_overdue_invoices"])
    (query,) = run.drafts["query_bill"]
    assert "£1,236.80" in query.body + query.subject and "£391.30" in query.body + query.subject
    gemini_drafts = sum(d.source == "gemini" for ds in run.drafts.values() for d in ds)
    print(f"{gemini_drafts} of 5 drafts written by Gemini; analysis took {run.seconds} s")
    assert run.seconds <= 10, f"analysis took {run.seconds} s (NFR3 target is about 10 s)"


@needs_voice
def test_live_elevenlabs_connection():
    ok, message = voice.ping()
    assert ok, message


@needs_voice
def test_live_tc4_briefing_starts_within_5_seconds(data):
    run = llm.analyse(*data)
    print("\nBriefing:", run.briefing)
    started = time.monotonic()
    audio, how = voice.speak(run.briefing)
    seconds = time.monotonic() - started
    print(f"Voice ready in {seconds:.1f} s ({len(audio):,} bytes)")
    assert how == "live"
    assert seconds <= 5, f"voice took {seconds:.1f} s (NFR3 target is about 5 s)"


@needs_voice
def test_live_tc4_spoken_question_round_trip(data):
    """Speak the question with ElevenLabs, transcribe it back, answer it from the data."""
    run = llm.analyse(*data)
    recording = voice.text_to_speech("Who owes me money?")
    question = voice.transcribe(recording, "question.mp3", "audio/mpeg")
    print("\nHeard:", question)
    assert "owe" in question.lower()
    text, source, note = voice.answer_question(question, llm.agent_facts(*data), run.plan,
                                               use_gemini=bool(config.gemini_api_key()))
    print("Answer:", text, f"({source}: {note})")
    assert names_everyone_who_owes(text)
