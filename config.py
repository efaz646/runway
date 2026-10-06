"""Runway settings: dates, thresholds, categories, model names and voice settings.

API keys are read, in order, from environment variables, a local .env file, and
Streamlit secrets (.streamlit/secrets.toml locally, or the Secrets box on Streamlit
Community Cloud). A .env file never overrides a variable that is already set.
Keys are never written into code.
"""

import os
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SECRETS_FILES = (BASE_DIR / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml")


ENV_FILE = BASE_DIR / ".env"
KEY_NAMES = ("GEMINI_API_KEY", "ELEVENLABS_API_KEY")


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """KEY=VALUE lines from .env (no extra dependency). utf-8-sig: Notepad may add a BOM."""
    path = path or ENV_FILE
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    return values


# Other settings in .env (for example GEMINI_MODEL) apply at start-up. API keys are
# read fresh each time instead, so editing .env needs only a page refresh.
for _key, _value in read_env_file().items():
    if _key not in KEY_NAMES:
        os.environ.setdefault(_key, _value)


def _secret(name: str) -> str | None:
    """Environment variable, then .env (read now), then Streamlit secrets."""
    value = os.environ.get(name, "").strip() or read_env_file().get(name, "").strip()
    if value:
        return value
    if any(path.exists() for path in SECRETS_FILES):  # only touch st.secrets when a secrets file exists
        try:
            import streamlit as st

            return str(st.secrets.get(name, "")).strip() or None
        except Exception:
            return None
    return None


def gemini_api_key() -> str | None:
    return _secret("GEMINI_API_KEY")


def elevenlabs_api_key() -> str | None:
    return _secret("ELEVENLABS_API_KEY")


# ---------------------------------------------------------------- sample data
BANK_STATEMENT_CSV = BASE_DIR / "bank_statement.csv"
INVOICES_CSV = BASE_DIR / "invoices.csv"

BUSINESS_NAME = "Bradford Auto Care"
OWNER_NAME = "Dave"
SAMPLE_DATA_NOTICE = (
    "SAMPLE DATA. Bradford Auto Care and every customer and supplier here are made up. "
    "No bank is connected, and no email is ever sent."
)

# "Today" in Dave's sample data. Fixed so the known answers never drift.
# Forecast week 1 starts on this date (a Monday); the statement ends the day before.
AS_OF_DATE = date(2026, 10, 5)

# ------------------------------------------------------------------ forecast
FORECAST_WEEKS = 8
AVERAGE_WINDOW_WEEKS = 8  # weekly averages use the last 8 full weeks of the statement

# First matching keyword wins, so the order matters.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("opening_balance", ["BALANCE BROUGHT FORWARD", "BALANCE B/F"]),
    ("card_takings", ["TILLPOINT SETTLEMENT"]),
    ("customer_payment", ["FPI "]),
    ("wages", ["WAGES"]),
    ("rent", ["CANAL ROAD ESTATES"]),
    ("electricity", ["NORTHERN GRID ENERGY"]),
    ("parts_account", ["PENNINE MOTOR FACTORS"]),
    ("vat", ["HMRC VAT"]),
    ("equipment_finance", ["AIRE VALLEY FINANCE"]),
    ("insurance", ["WHITE ROSE INSURANCE"]),
    ("phone", ["DALES TELECOM"]),
]
DEFAULT_CATEGORY = "other"

# Plain-English names shown to Dave.
CATEGORY_LABELS = {
    "card_takings": "Card takings",
    "customer_payment": "Customer invoice payments",
    "wages": "Wages",
    "rent": "Rent (Canal Road Estates)",
    "electricity": "Electricity (Northern Grid Energy)",
    "parts_account": "Parts account (Pennine Motor Factors)",
    "vat": "VAT (HMRC)",
    "equipment_finance": "Equipment finance (Aire Valley Finance)",
    "insurance": "Insurance (White Rose Insurance)",
    "phone": "Phone and internet (Dales Telecom)",
    "other": "Other spending",
    "invoice_due": "Invoices due in",
}

# How each regular payment repeats. Its next date follows the last one in the statement,
# and its amount is the median of recent payments (so a one-off spike is not repeated).
RECURRING_CADENCE = {
    "wages": "weekly",
    "rent": "monthly",
    "electricity": "monthly",
    "parts_account": "monthly",
    "equipment_finance": "monthly",
    "insurance": "monthly",
    "phone": "monthly",
    "vat": "quarterly",
}
RECURRING_LOOKBACK = {"weekly": 8, "monthly": 6, "quarterly": 4}  # occurrences used for the median

# Spread evenly across forecast weeks, using the weekly average of the last 8 weeks.
WEEKLY_AVERAGE_CATEGORIES = ["card_takings", "other"]

# Payments that can reasonably be moved by agreement (used for "delay one payment").
DEFERRABLE_CATEGORIES = ["parts_account"]
DELAY_DAYS = 14  # how far Runway suggests moving that payment

# ---------------------------------------------------------------- bill checker
SUPPLIER_BILL_CATEGORIES = ["electricity", "parts_account", "insurance", "phone"]
SPIKE_RATIO = 1.5  # a month is a spike if it is over 1.5x the usual level (median of the other months)

# ----------------------------------------------------------- late payment chaser
# Tone by days overdue: up to 14 friendly, up to 44 firm, 45 or more final notice.
TONE_FRIENDLY_MAX_DAYS = 14
TONE_FIRM_MAX_DAYS = 44
REMINDER_PAY_WITHIN_DAYS = 7

# Made-up supplier contacts for the drafts (".example" addresses can never be delivered).
SUPPLIERS = {
    "electricity": {"name": "Northern Grid Energy", "bill": "Electricity", "email": "billing@northerngridenergy.example"},
    "parts_account": {"name": "Pennine Motor Factors", "bill": "Parts account", "email": "accounts@penninemotorfactors.example"},
    "insurance": {"name": "White Rose Insurance", "bill": "Insurance", "email": "service@whiteroseinsurance.example"},
    "phone": {"name": "Dales Telecom", "bill": "Phone and internet", "email": "billing@dalestelecom.example"},
}

# -------------------------------------------------------------------- Gemini
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
GEMINI_THINKING_LEVEL = "LOW"  # keep replies fast for the demo; set to None if the model rejects it
GEMINI_TIMEOUT_SECONDS = 15
AGENT_MAX_TURNS = 5  # look up facts, submit plan, fix plan if sent back
AGENT_DEADLINE_SECONDS = 25  # after this, Runway stops waiting and falls back
DRAFTS_DEADLINE_SECONDS = 20
CACHE_FILE = BASE_DIR / "cache" / "last_agent_run.json"  # last verified agent run, used if Gemini fails live
CACHE_VERSION = "3"

# ---------------------------------------------------------------- ElevenLabs
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # "George", a British premade voice
ELEVENLABS_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")  # lowest-latency TTS model
ELEVENLABS_STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2")
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_TIMEOUT_SECONDS = 15
