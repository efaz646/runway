"""Runway: one-screen Streamlit app for the live demo.

Code calculates every figure. Gemini acts as the agent: it looks up the facts,
decides the actions and writes every draft, and Runway checks each number against
the data. Nothing is ever sent.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import time

import altair as alt
import pandas as pd
import streamlit as st

import bills
import chaser
import config
import forecast
import llm
import voice
from forecast import money, short_date

st.set_page_config(page_title="Runway · Bradford Auto Care", page_icon="📉", layout="wide")

GREEN, RED, GREY = "#2f7d5b", "#c0392b", "#8a99a8"
ACTION_ICON = {"chase_overdue_invoices": "📨", "query_bill": "⚡", "delay_payment": "📅"}


@st.cache_data(show_spinner=False)
def load_everything():
    statement, invoices, fc = forecast.load_and_forecast()
    raw_preview = config.BANK_STATEMENT_CSV.read_text(encoding="utf-8").splitlines()[:16]
    monthly = bills.monthly_supplier_costs(statement)
    unpaid = chaser.unpaid_invoices(invoices, fc.as_of)
    return statement, invoices, fc, raw_preview, monthly, unpaid


# ---------------------------------------------------------------------- charts


def cash_chart(weeks: pd.DataFrame) -> alt.LayerChart:
    df = weeks[["week", "week_label", "closing_balance"]].copy()
    df["status"] = df["closing_balance"].map(lambda v: "Below zero" if v < 0 else "Above zero")
    df["balance"] = df["closing_balance"].map(money)
    low, high = float(df["closing_balance"].min()), float(df["closing_balance"].max())
    domain = [min(low * 1.4, -500.0), max(high * 1.15, 500.0)]  # headroom so labels never sit on the axis
    x = alt.X("week_label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=11))
    bars = (
        alt.Chart(df)
        .mark_bar(size=38, cornerRadiusEnd=3)
        .encode(
            x=x,
            y=alt.Y("closing_balance:Q", title="Cash at end of week (£)", axis=alt.Axis(format=",.0f"),
                    scale=alt.Scale(domain=domain, nice=False)),
            color=alt.Color("status:N", scale=alt.Scale(domain=["Above zero", "Below zero"], range=[GREEN, RED]), legend=None),
            tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("balance:N", title="Cash at end of week")],
        )
    )
    above = bars.transform_filter("datum.closing_balance >= 0").mark_text(dy=-9, fontSize=11, color=GREEN).encode(text="balance:N")
    below = bars.transform_filter("datum.closing_balance < 0").mark_text(dy=11, fontSize=11, color=RED, fontWeight="bold").encode(text="balance:N")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#444", strokeWidth=1.5).encode(y="y:Q")
    layers = bars + zero + above + below
    short = df[df["closing_balance"] < 0]
    if not short.empty:  # flag the first week below zero
        flag = short.head(1).assign(y=0, note="Runs short")
        layers += alt.Chart(flag).mark_text(dy=-10, fontSize=13, fontWeight="bold", color=RED).encode(x=x, y="y:Q", text="note:N")
    return layers.properties(height=340)


def bill_chart(one_supplier: pd.DataFrame) -> alt.LayerChart:
    df = one_supplier[["month", "amount", "usual", "is_spike"]].copy()
    df["status"] = df["is_spike"].map(lambda s: "Spike" if s else "Normal")
    df["label"] = df.apply(lambda r: f"{money(r.amount)} · {r.amount / r.usual:.1f}× usual" if r.is_spike else money(r.amount), axis=1)
    usual = float(df["usual"].iloc[0])
    top = max(float(df["amount"].max()), usual) * 1.2
    x = alt.X("month:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=11))
    bars = (
        alt.Chart(df)
        .mark_bar(size=38, cornerRadiusEnd=3)
        .encode(
            x=x,
            y=alt.Y("amount:Q", title="Monthly bill (£)", axis=alt.Axis(format=",.0f"), scale=alt.Scale(domain=[0, top], nice=False)),
            color=alt.Color("status:N", scale=alt.Scale(domain=["Normal", "Spike"], range=[GREY, RED]), legend=None),
            tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("label:N", title="Bill")],
        )
    )
    df["amount_text"] = df["amount"].map(money)
    spike_label = (
        alt.Chart(df[df["is_spike"]])
        .mark_text(dy=-9, fontSize=12, fontWeight="bold", color=RED)
        .encode(x=x, y="amount:Q", text="amount_text:N")
    )
    rule = alt.Chart(pd.DataFrame({"y": [usual]})).mark_rule(color="#333", strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q")
    title = alt.TitleParams(
        f"{one_supplier['bill'].iloc[0]} bills by month",
        subtitle=f"Dashed line: usual level, {money(usual)} a month" + (" · Red bar: spike" if df["is_spike"].any() else ""),
        anchor="start", fontSize=14, subtitleFontSize=12, subtitleColor="#444",
    )
    return (bars + rule + spike_label).properties(height=300, title=title)


# ---------------------------------------------------------------------- dialog


@st.dialog("Draft email", width="large")
def show_drafts(kind: str) -> None:
    run: llm.AgentRun = st.session_state.agent_run
    drafts = run.drafts.get(kind, [])
    st.info("Draft only. Runway never sends anything. Copy it into your own email if you're happy with it.", icon="📭")
    boxes = st.tabs([d.label for d in drafts]) if len(drafts) > 1 else [st.container()]
    for box, d in zip(boxes, drafts):
        with box:
            st.markdown(f"**To:** {d.to.replace('<', '(').replace('>', ')')}  \n**Subject:** {d.subject}")
            st.code(d.body, language=None, wrap_lines=True)
            st.caption(("✅ " if d.source == "gemini" else "📝 ") + d.note)


def open_button(kind: str, n: int, key: str) -> None:
    label = {
        "chase_overdue_invoices": f"Open {n} draft reminder{'s' if n != 1 else ''}",
        "query_bill": "Open draft query email",
        "delay_payment": "Open draft request",
    }[kind]
    if st.button(label, key=key, icon="✉️", disabled=n == 0):
        show_drafts(kind)


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Connections")
    st.write(f"Gemini key: {'✅ set' if config.gemini_api_key() else '❌ missing'}")
    st.write(f"ElevenLabs key: {'✅ set' if config.elevenlabs_api_key() else '❌ missing'}")
    if st.button("Test both connections"):
        with st.spinner("Calling Gemini and ElevenLabs…"):
            for ok, message in (llm.ping(), voice.ping()):
                (st.success if ok else st.error)(message)
    if not (config.gemini_api_key() and config.elevenlabs_api_key()):
        st.caption(f"Add the keys to this file, save it, then refresh this page:  \n`{config.ENV_FILE}`")
        if (config.BASE_DIR / ".env.txt").exists():
            st.warning("Found **.env.txt**. Windows added .txt to the name, so rename it to **.env**.")
        elif not config.ENV_FILE.exists():
            st.warning("There is no .env file in the app folder yet.")
    st.caption("Keys come from environment variables or a local .env file. They are never stored in the code.")
    st.divider()
    st.subheader("Rehearsal")
    no_gemini = st.toggle("Pretend Gemini is down", help="Shows what the audience sees if Gemini fails during the demo. Press 'Run Runway's agent again' after switching.")
    no_voice = st.toggle("Pretend ElevenLabs is down", help="Shows the text briefing (or the saved recording) instead of live voice.")
    if st.button("Run Runway's agent again"):
        for key in ("agent_run", "briefing_audio", "answer"):
            st.session_state.pop(key, None)

# ---------------------------------------------------------------------- header
st.title("Runway")
st.markdown("**Businesses don't fail suddenly. They fail because problems are spotted too late.** "
            "Runway watches the money, warns early, and does the admin to fix it.")
st.warning(config.SAMPLE_DATA_NOTICE, icon="🧪")

if not st.session_state.get("loaded"):
    if st.button(f"Load {config.OWNER_NAME}'s sample data ({config.BUSINESS_NAME})", type="primary", icon="📂"):
        st.session_state.loaded = True
        st.session_state.pop("agent_run", None)
        st.rerun()
    st.stop()

statement, invoices, fc, raw_preview, monthly, unpaid = load_everything()

with st.expander(f"{config.OWNER_NAME}'s bank export, as downloaded (messy)"):
    st.code("\n".join(raw_preview) + "\n…", language=None)
    st.caption(f"{len(statement)} lines from {short_date(statement['date'].min())} to {short_date(statement['date'].max())}, "
               "newest first, with mixed date and amount formats. Runway cleans it before doing any sums.")

# ---------------------------------------------------------------- 1. forecast
st.header(f"Cash runway for {config.BUSINESS_NAME}")
st.caption(f"Forecast from {short_date(fc.as_of)} for the next {len(fc.weeks)} weeks. Every figure is calculated from the data.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cash today", money(fc.current_balance))
c2.metric("Weeks of cash left", f"{fc.weeks_of_cash_left}")
if fc.runs_short:
    c3.metric("Runs short", f"Week {fc.shortfall_week}", f"w/c {short_date(fc.shortfall_week_start)}", delta_color="off")
else:
    c3.metric("Runs short", "Not in 8 weeks")
c4.metric("Lowest point", money(fc.lowest_balance), f"Week {fc.lowest_week}", delta_color="off")

if fc.runs_short:
    st.error(f"**⚠️ {fc.warning}** (week of {short_date(fc.shortfall_week_start)}). "
             f"The balance that week is forecast to be **{money(fc.shortfall_balance)}**.")
else:
    st.success(fc.warning)

left, right = st.columns([3, 2], gap="large")

with left:
    st.altair_chart(cash_chart(fc.weeks), width="stretch")
    with st.expander("How this forecast is worked out"):
        st.markdown(
            "- **Cash today** is the sum of every line in the statement.\n"
            "- **Regular payments** (wages, rent, bills, VAT) repeat on their usual pattern, at their usual amount.\n"
            "- **Card takings and small spending** use the average of the last 8 weeks.\n"
            "- **Invoices not yet due** are expected on their due date. **Overdue invoices are not counted**, because they are already late."
        )
        table = fc.flows.assign(
            Date=fc.flows["date"].map(short_date), Week=fc.flows["week"], Item=fc.flows["label"], Amount=fc.flows["amount"].map(money)
        )[["Week", "Date", "Item", "Amount"]]
        st.dataframe(table, hide_index=True, width="stretch", height=260)

with right:
    st.subheader("What Runway recommends")
    if "agent_run" not in st.session_state:
        with st.status("Runway's agent is checking the numbers…", expanded=True) as status:
            run = llm.analyse(statement, invoices, fc, progress=status.write, allow_gemini=not no_gemini)
            status.update(label=f"Checked in {run.seconds} seconds", state="complete", expanded=False)
        st.session_state.agent_run = run
    run: llm.AgentRun = st.session_state.agent_run

    if run.source == "gemini":
        st.caption(f"🤖 Decided by Gemini ({run.model}) in {run.seconds} s. Every figure checked against the data.")
    elif run.source == "cache":
        st.warning(f"Gemini didn't respond, so this is Runway's last verified analysis (saved {run.created_at}).", icon="💾")
    else:
        st.warning("Gemini is unavailable, so Runway is showing its built-in plan. Every figure is still calculated from the data.", icon="⚠️")

    st.markdown(f"**{run.plan['headline']}**")
    for action in run.plan["actions"]:
        with st.container(border=True):
            st.markdown(f"**{action['priority']}. {ACTION_ICON[action['type']]} {action['title']}**")
            st.write(action["why"])
            open_button(action["type"], len(run.drafts.get(action["type"], [])), key=f"open_{action['type']}")

st.divider()

# -------------------------------------------------------------- 2. bill checker
st.header("Bill checker")
bill_left, bill_right = st.columns([3, 2], gap="large")
spikes = monthly[monthly["is_spike"]]
with bill_left:
    names = list(dict.fromkeys(monthly["bill"]))
    default = names.index(spikes["bill"].iloc[0]) if not spikes.empty else 0
    chosen = st.selectbox("Supplier", names, index=default, label_visibility="collapsed")
    one = monthly[monthly["bill"] == chosen]
    st.altair_chart(bill_chart(one), width="stretch")
    supplier = one["supplier"].iloc[0]
    if one["is_spike"].any():
        s = one[one["is_spike"]].iloc[0]
        st.caption(f"{supplier}: {s['month']} was {money(s['amount'])}, against a usual {money(s['usual'])} a month. Other months are normal.")
    else:
        st.caption(f"{supplier}: no spikes. Every month is close to the usual {money(one['usual'].iloc[0])}.")

with bill_right:
    if spikes.empty:
        st.success("No supplier bill has spiked.")
    else:
        s = spikes.iloc[0]
        who = "Gemini" if run.source in ("gemini", "cache") else "Runway"
        st.subheader(f"{who}'s suggestions")
        st.caption(f"For the {s['month']} {s['bill'].lower()} bill from {s['supplier']}")
        for suggestion in run.plan["bill_suggestions"]:
            st.markdown(f"- {suggestion}")
        open_button("query_bill", len(run.drafts.get("query_bill", [])), key="open_query_bill_bills")

st.divider()

# ------------------------------------------- 3. who owes you + voice briefing
owe_left, voice_right = st.columns([3, 2], gap="large")
with owe_left:
    st.header("Who owes you money")
    table = pd.DataFrame(
        {
            "Customer": unpaid["customer"],
            "Invoice": unpaid["invoice_number"],
            "Amount": unpaid["amount"].map(money),
            "Due": unpaid["due_date"].map(short_date),
            "Status": unpaid.apply(
                lambda r: f"{r.days_overdue} days overdue ({r.tone})" if r.overdue else f"Due in {-r.days_overdue} days", axis=1
            ),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    overdue = unpaid[unpaid["overdue"]]
    m1, m2 = st.columns(2)
    m1.metric("Overdue", money(overdue["amount"].sum()), f"{len(overdue)} invoices", delta_color="off")
    m2.metric("Owed in total", money(unpaid["amount"].sum()), f"{unpaid['customer'].nunique()} customers", delta_color="off")
    open_button("chase_overdue_invoices", len(run.drafts.get("chase_overdue_invoices", [])), key="open_chase_owed")

with voice_right:
    st.header("Morning briefing")
    if st.button("Play the 30-second briefing", type="primary", icon="🔊"):
        started = time.monotonic()
        try:
            audio, how = voice.speak(run.briefing, allow=not no_voice)
            st.session_state.briefing_audio = {"audio": audio, "how": how, "seconds": round(time.monotonic() - started, 1), "fresh": True}
        except voice.VoiceUnavailable as exc:
            st.session_state.briefing_audio = {"audio": None, "how": str(exc), "seconds": 0, "fresh": True}
    played = st.session_state.get("briefing_audio")
    if played and played["audio"]:
        st.audio(played["audio"], format="audio/mpeg", autoplay=played.pop("fresh", False))
        st.caption(f"🔊 ElevenLabs voice, ready in {played['seconds']} s." if played["how"] == "live"
                   else "💾 ElevenLabs is unavailable, so this is the saved recording of the same briefing.")
    elif played:
        st.info("The voice is unavailable right now, so here is the briefing as text.", icon="📝")
    with st.expander("Briefing text", expanded=bool(played and not played["audio"])):
        st.write(run.briefing)
        st.caption("✅ Written by Gemini, figures checked against the data." if run.briefing_source == "gemini"
                   else "📝 Template briefing, written from the same figures.")

    st.subheader("Ask Runway")
    recording = st.audio_input("Ask a question out loud", key="question_audio")
    typed = st.text_input("Or type a question", placeholder="Who owes me money?", key="typed_question")
    b1, b2 = st.columns(2)
    ask_typed = b1.button("Ask", icon="💬", width="stretch")
    ask_quick = b2.button("Who owes me money?", width="stretch")

    question, heard = None, False
    if recording is not None:
        digest = hashlib.sha256(recording.getvalue()).hexdigest()
        if digest != st.session_state.get("last_recording"):
            st.session_state.last_recording = digest
            try:
                with st.spinner("Listening…"):
                    question, heard = voice.transcribe(recording.getvalue(), allow=not no_voice), True
            except voice.VoiceUnavailable:
                st.warning("Runway couldn't make out the recording, so please type the question instead.", icon="🎙️")
    if ask_typed and typed.strip():
        question = typed.strip()
    elif ask_quick:
        question = "Who owes me money?"

    if question:
        with st.spinner("Runway is checking the figures…"):
            facts = llm.agent_facts(statement, invoices, fc)
            use_gemini = run.source == "gemini" and not no_gemini
            text, source, _ = voice.answer_question(question, facts, run.plan, use_gemini=use_gemini)
            try:
                audio, _how = voice.speak(text, allow=not no_voice)
            except voice.VoiceUnavailable:
                audio = None
        st.session_state.answer = {"question": question, "heard": heard, "text": text, "source": source, "audio": audio, "fresh": True}

    answer = st.session_state.get("answer")
    if answer:
        st.markdown(f"**{'You asked' if answer['heard'] else 'Question'}:** {answer['question']}")
        st.success(answer["text"], icon="💬")
        if answer["audio"]:
            st.audio(answer["audio"], format="audio/mpeg", autoplay=answer.pop("fresh", False))
        st.caption(("✅ Answered by Gemini from the data, figures checked." if answer["source"] == "gemini"
                    else "📝 Answered from the data by Runway's template.") + ("" if answer["audio"] else " Voice unavailable."))

st.divider()
with st.expander("What Runway's agent did"):
    for i, line in enumerate(run.log, 1):
        st.markdown(f"{i}. {line}")
