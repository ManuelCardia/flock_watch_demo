"""
Flock Watch - interactive demo (Streamlit).

Run:  pip install streamlit scikit-learn numpy pandas altair
      streamlit run app.py
Keep app.py, flock_watch_core_v3.py and .streamlit/config.toml in the same folder.

WHAT THIS IS: a presentation demo built on a SIMULATION. Sensor readings, health events,
thresholds, rewards and every metric shown are simulated placeholders. Nothing here is
evidence that the system detects disease on a real farm.

Structure
  1. Theme + CSS
  2. Small helpers (time, plain-language explanations, state of each animal)
  3. Setup wizard (landing screen)
  4. Sidebar demo controls (presentation helpers, not part of the farmer product)
  5. Tab 1  Monitoring: filtered anomaly feed + embedded animal drawer with feedback form
  6. Tab 2  Feedback history and performance
  7. Tab 3  Welfare and values
The mock backend is flock_watch_core_v3.py (per-animal baselines, Isolation Forest score,
quantile thresholds, k-of-n persistence, tiers, exposure, conservative bandit, audits).
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math

import altair as alt
import pandas as pd
import streamlit as st

import flock_watch_core_v3 as core

# ---------------------------------------------------------------------------
# 1. Theme
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Flock Watch", page_icon="🐑", layout="wide", initial_sidebar_state="collapsed")

GREEN, MINT = "#2F855A", "#E6FFFA"
BG, GRAY = "#F7FAFC", "#EDF2F7"
INK, MUTED = "#2D3748", "#5A6678"
AMBER, RED = "#DD6B20", "#E53E3E"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Figtree:wght@400;500;600&display=swap');
:root {{ color-scheme: light; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background:{BG}; color:{INK}; font-family:'Figtree', system-ui, sans-serif; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1280px; }}
h1,h2,h3,h4 {{ font-family:'Fraunces', Georgia, serif !important; color:{INK} !important; font-weight:600 !important; letter-spacing:-0.01em; }}
p, li, label, span, div {{ color:{INK}; }}
a {{ color:{GREEN}; }}
[data-testid="stSidebar"] {{ background:{GRAY}; }}
[data-testid="stSidebar"] * {{ color:{INK}; }}

.fw-muted {{ color:{MUTED} !important; font-size:0.92rem; line-height:1.45; }}
.fw-card {{ background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:1.1rem 1.3rem; box-shadow:0 1px 3px rgba(45,55,72,.06), 0 6px 18px rgba(45,55,72,.04); }}
.fw-card.mint {{ background:{MINT}; border-color:#B2F5EA; }}
.fw-card h4 {{ margin:0 0 .35rem 0; font-size:1.08rem; }}
.fw-hero {{ background:linear-gradient(135deg,#fff 0%,{MINT} 140%); border:1px solid #CBE9DF; border-radius:18px; padding:1.5rem 1.7rem; margin-bottom:1.1rem; }}
.fw-hero h1 {{ margin:0; font-size:2rem; }}
.fw-chip {{ display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.82rem; font-weight:600; margin:0 .35rem .3rem 0; border:1px solid transparent; }}
.fw-chip.ok {{ background:{MINT}; color:#276749; border-color:#B2F5EA; }}
.fw-chip.warn {{ background:#FFFAF0; color:#9C4221; border-color:#FBD38D; }}
.fw-chip.alert {{ background:#FFF5F5; color:#C53030; border-color:#FEB2B2; }}
.fw-chip.neutral {{ background:{GRAY}; color:{INK}; border-color:#E2E8F0; }}

.fw-stat {{ background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:.9rem 1.1rem; height:100%; }}
.fw-stat .n {{ font-family:'Fraunces',serif; font-size:2rem; line-height:1.1; font-weight:600; }}
.fw-stat .l {{ color:{MUTED}; font-size:.9rem; }}
.fw-stat.amber {{ border-left:5px solid {AMBER}; }} .fw-stat.red {{ border-left:5px solid {RED}; }}
.fw-stat.green {{ border-left:5px solid {GREEN}; }} .fw-stat.slate {{ border-left:5px solid #A0AEC0; }}

[class*="st-key-card_"] {{ background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:1rem 1.2rem .6rem 1.2rem; margin-bottom:.8rem;
    box-shadow:0 1px 3px rgba(45,55,72,.06), 0 6px 18px rgba(45,55,72,.04); }}
[class*="st-key-card_report"] {{ border-left:6px solid {RED}; }}
[class*="st-key-card_isolate"], [class*="st-key-card_inspect"] {{ border-left:6px solid {AMBER}; }}
[class*="st-key-card_watch"] {{ border-left:6px solid #F6AD55; }}
[class*="st-key-card_sel"] {{ box-shadow:0 0 0 2px {GREEN}, 0 6px 18px rgba(47,133,90,.18); }}
.st-key-drawer {{ background:#fff; border:1px solid #CBE9DF; border-radius:16px; padding:1.3rem 1.5rem; box-shadow:0 8px 28px rgba(47,133,90,.10); }}
.st-key-drawer table.fw-table {{ font-size:.82rem; }}
.st-key-drawer table.fw-table td, .st-key-drawer table.fw-table th {{ padding:.35rem .3rem; }}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ display:none; }}
.fw-card.tall {{ min-height:10.2rem; }}
.fw-id {{ font-family:'Fraunces',serif; font-size:1.25rem; font-weight:600; }}
.fw-tier {{ display:inline-block; padding:.15rem .6rem; border-radius:8px; font-size:.8rem; font-weight:600; float:right; }}
.fw-tier.report {{ background:#FFF5F5; color:#C53030; border:1px solid #FEB2B2; }}
.fw-tier.isolate, .fw-tier.inspect {{ background:#FFFAF0; color:#9C4221; border:1px solid #FBD38D; }}
.fw-tier.watch {{ background:#FFFFF0; color:#744210; border:1px solid #F6E05E; }}
.fw-tier.official {{ background:{GRAY}; color:{INK}; border:1px solid #CBD5E0; }}

.fw-ladder {{ display:flex; gap:.4rem; margin:.4rem 0 .8rem 0; }}
.fw-ladder div {{ flex:1; padding:.5rem .6rem; border-radius:10px; background:{GRAY}; color:{MUTED}; font-size:.85rem; border:1px solid transparent; }}
.fw-ladder div b {{ display:block; color:inherit; }}
.fw-ladder .on.watch {{ background:#FFFFF0; border-color:#F6E05E; color:#744210; }}
.fw-ladder .on.isolate {{ background:#FFFAF0; border-color:{AMBER}; color:#9C4221; }}
.fw-ladder .on.report {{ background:#FFF5F5; border-color:{RED}; color:#C53030; }}

table.fw-table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
table.fw-table th {{ text-align:left; color:{MUTED}; font-weight:600; padding:.45rem .5rem; border-bottom:1px solid #E2E8F0; }}
table.fw-table td {{ padding:.5rem .5rem; border-bottom:1px solid #EDF2F7; vertical-align:top; }}
table.fw-table td.flag {{ color:#9C4221; font-weight:600; background:#FFFAF0; }}

.fw-dots {{ display:flex; flex-wrap:wrap; gap:5px; margin:.3rem 0 .6rem 0; }}
.fw-pen {{ margin-right:1.4rem; margin-bottom:.4rem; }}
.fw-pen .pl {{ font-size:.8rem; color:{MUTED}; margin-bottom:2px; }}
.dot {{ width:11px; height:11px; border-radius:50%; background:#9AD5B5; display:inline-block; }}
.dot.watch {{ background:#fff; border:2.5px solid {AMBER}; box-sizing:border-box; }}
.dot.isolate {{ background:{AMBER}; }}
.dot.inspect {{ background:{AMBER}; border-radius:2px; transform:rotate(45deg); }}
.dot.report {{ background:{RED}; width:14px; height:14px; border-radius:3px; box-shadow:0 0 0 3px rgba(229,62,62,.2); }}
.dot.official {{ background:#4A5568; border-radius:2px; }}
.fw-legend span {{ margin-right:1.1rem; font-size:.85rem; color:{MUTED}; }}
.fw-legend .dot {{ margin-right:.35rem; vertical-align:-1px; }}

.fw-steps {{ display:flex; gap:.5rem; margin:.2rem 0 1.2rem 0; }}
.fw-steps div {{ flex:1; padding:.55rem .8rem; border-radius:10px; background:{GRAY}; color:{MUTED}; font-size:.9rem; border-bottom:4px solid transparent; }}
.fw-steps div.done {{ background:{MINT}; color:#276749; }} .fw-steps div.cur {{ background:#fff; color:{INK}; border-bottom-color:{GREEN}; font-weight:600; box-shadow:0 1px 4px rgba(45,55,72,.1); }}

.stButton > button {{ border-radius:10px; font-weight:600; border:1px solid #CBD5E0; background:#fff; color:{INK}; }}
.stButton > button:hover {{ border-color:{GREEN}; color:{GREEN}; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background:{GREEN}; border-color:{GREEN}; color:#fff; }}
.stButton > button[kind="primary"]:hover {{ background:#276749; color:#fff; }}
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primary"] p, .stButton > button[kind="primary"]:hover p {{ color:#fff !important; }}
table.fw-table, table.fw-table td, table.fw-table th {{ border-left:none; border-right:none; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:.5rem 1rem; }}
.stTabs [aria-selected="true"] {{ color:{GREEN} !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

TIER_LABEL = {"watch": "Watchlist", "isolate": "Soft isolation", "inspect": "Inspection requested",
              "report": "Report suspicion", "official": "Official pathway"}
OUTCOME_CHOICES = {
    "🩺 Vet confirmed a health problem": (core.OUT_POS, core.EV_VET),
    "🩺 Vet found no problem": (core.OUT_NEG, core.EV_VET),
    "🏛 Reference laboratory confirmed": (core.OUT_POS, core.EV_LAB),
    "👁 I inspected: animal looks healthy (false alarm)": (core.OUT_NEG, core.EV_FARMER),
    "🧪 Pen-side test negative, no other signs (presumptive)": (core.OUT_NEG, core.EV_PENTEST),
    "🧪 Pen-side test positive (presumptive)": (core.OUT_POS, core.EV_PENTEST),
    "⚙ Sensor or identification fault": (core.OUT_FAULT, core.EV_FARMER),
    "❔ Still uncertain: keep this case open": (core.OUT_UNC, core.EV_FARMER),
}
EVIDENCE_ICON = {core.EV_VET: "🩺 Veterinary", core.EV_LAB: "🏛 Reference lab", core.EV_PENTEST: "🧪 Pen-side test",
                 core.EV_FARMER: "👁 Farmer", core.EV_AUDIT: "🎯 Random check"}
PENTEST_PATTERNS = {"Febrile infectious", "Dermatological infectious"}
DISPLAY_FEATURES = ["thermal", "gait", "cough", "speed", "lesion"]


def fw_() -> core.FlockWatch:
    return st.session_state.fw


def when(rnd: int, pass_s: float = 0.0) -> str:
    base = dt.datetime.combine(st.session_state.start_date, dt.time(6, 30))
    return (base + dt.timedelta(days=rnd, seconds=pass_s)).strftime("%a %d %b, %H:%M")


def esc(x) -> str:
    return html.escape(str(x))


def adjusted_values(fw: core.FlockWatch, raw: dict, ambient: float) -> dict:
    """Raw reading with the thermal value adjusted for ambient temperature."""
    v = dict(raw)
    v["thermal"] = raw["thermal"] - fw.beta_thermal * (ambient - core.T_REF)
    return v


def signal_phrases(fw: core.FlockWatch, aid: str, z: dict, vals: dict, thr: float) -> list:
    """Plain-language sentences for features whose z-score magnitude is >= thr, strongest first."""
    start = fw.start_mean.get(aid, {})
    out = []
    for f in sorted(z, key=lambda k: -abs(z[k])):
        if abs(z[f]) < thr or (f in ("gait", "cough", "lesion") and z[f] < 0):
            continue   # a decrease in asymmetry, coughing or lesions is not a warning sign
        s = start.get(f, vals[f])
        if f == "thermal":
            d = vals[f] - s
            out.append((f, f"Eye-region temperature is {abs(d):.1f} °C {'above' if d > 0 else 'below'} this animal's normal (adjusted for the weather)."))
        elif f == "gait":
            out.append((f, f"Walking is less even than usual: asymmetry {vals[f]:.0f}/100 against {s:.0f}/100 at baseline."))
        elif f == "cough":
            out.append((f, f"More coughing at the raceway: {vals[f]:.1f} events against {s:.1f} normally."))
        elif f == "speed":
            pct = (vals[f] / s - 1) * 100
            out.append((f, f"Moving {abs(pct):.0f}% {'slower' if pct < 0 else 'faster'} through the raceway than usual."))
        elif f == "lesion":
            out.append((f, f"Skin-lesion score is {vals[f]:.0f}/100 against {s:.0f}/100 at baseline."))
    return out


def short_signal(fw, aid, z, vals) -> str:
    ph = signal_phrases(fw, aid, z, vals, 1.5)
    if not ph:
        return "Several small deviations together."
    return ph[0][1].split(" (")[0]


def compute_states(fw: core.FlockWatch) -> dict:
    """aid -> (state, alert record or None). State in normal/watch/inspect/isolate/report/official."""
    open_by = {}
    for r in fw.alerts:
        if not r.resolved:
            open_by[r.animal_id] = r
    horizon = fw.round_no - fw.context.handling_interval + 1
    out = {}
    for aid, a in fw.animals.items():
        r = open_by.get(aid)
        if a.status == "official":
            out[aid] = ("official", None)
        elif r is not None:
            if r.suspicion:
                out[aid] = ("report", r)
            elif a.isolated:
                out[aid] = ("isolate", r)
            else:
                out[aid] = ("inspect", r)
        else:
            rd = fw.readings[aid][-1] if fw.readings[aid] else None
            if rd and rd["round"] >= horizon and rd["tier"] == 1:
                out[aid] = ("watch", None)
            else:
                out[aid] = ("normal", None)
    return out


def advance(days: int):
    fw = fw_()
    for _ in range(days):
        amb = 18 + 8 * math.sin(2 * math.pi * (fw.round_no + 1) / 90) + fw.rng.gauss(0, 3)
        fw.step(amb)
        if st.session_state.auto_audit and fw.round_no % 7 == 0:
            res = fw.run_audit()
            missed = [r for r in res if r["result"] != "healthy"]
            if missed:
                st.session_state.flash = f"Weekly random check found {len(missed)} case(s) the station had not flagged."


def inject_case():
    fw = fw_()
    cands = [a for a in fw.animals.values() if a.status == "active" and a.event is None and not a.isolated]
    if not cands:
        return None
    a = fw.rng.choice(cands)
    a.event = {"pattern": "Febrile infectious", "kind": "disease", "day": 0, "duration": 10,
               "treated_day": None, "start_round": fw.round_no}
    return a.id


def select(kind: str, key):
    st.session_state.sel = (kind, key)


def close_drawer():
    st.session_state.sel = None


# ---------------------------------------------------------------------------
# 3. Setup wizard
# ---------------------------------------------------------------------------

def hardware_plan(n: int, raceways: int) -> dict:
    sec_per_sheep = 7.0   # planning assumption: ~4-10 s per animal at the station
    minutes_single = n * sec_per_sheep / 60.0
    rec = max(1, math.ceil(minutes_single / 45.0))   # planning rule of thumb: keep a full pass under ~45 min
    return {"stations": raceways, "thermal": 2 * raceways, "rgbd": raceways, "mics": raceways,
            "edge": raceways, "minutes": minutes_single / raceways, "recommended": rec,
            "pen_capacity": max(4, math.ceil(0.05 * n))}


def wizard():
    s = st.session_state
    steps = ["Farm and infrastructure", "Cold-start baseline", "Hardware tiers", "Privacy and start"]
    cur = s.wstep
    chips = "".join(f'<div class="{"done" if i < cur else "cur" if i == cur else ""}">{i + 1}. {t}</div>'
                    for i, t in enumerate(steps))
    st.markdown(
        '<div class="fw-hero"><h1>Flock Watch</h1>'
        '<p class="fw-muted" style="margin:.3rem 0 0 0;font-size:1.02rem">A decision-support system that watches the flock as it walks through '
        'a raceway, so you only need to look closely at the few animals that need it. It advises; you decide.</p></div>'
        f'<div class="fw-steps">{chips}</div>', unsafe_allow_html=True)

    if cur == 0:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown("#### Your farm")
            s.cfg["farm_name"] = st.text_input("Farm name", s.cfg["farm_name"])
            s.cfg["n"] = st.slider("Flock size (sheep)", 50, 400, s.cfg["n"], step=10)
            s.cfg["raceways"] = st.number_input("Central passage raceways", 1, 4, s.cfg["raceways"])
            s.cfg["system"] = st.radio(
                "How are the sheep handled?",
                [core.SYSTEM_INTENSIVE, core.SYSTEM_EXTENSIVE],
                index=[core.SYSTEM_INTENSIVE, core.SYSTEM_EXTENSIVE].index(s.cfg["system"]),
                help="Extensive flocks only pass the station at handling rounds, so detection is slower by design.")
            if s.cfg["system"] == core.SYSTEM_EXTENSIVE:
                st.info("Extensive pasture: sheep pass the station about every 4th day in this demo. "
                        "Expect slower detection than on a daily raceway.")
        with c2:
            p = hardware_plan(s.cfg["n"], s.cfg["raceways"])
            ok = s.cfg["raceways"] >= p["recommended"]
            st.markdown("#### Recommended fixed hardware")
            st.markdown(
                f'<div class="fw-card mint"><table class="fw-table">'
                f'<tr><td>Passage stations</td><td><b>{p["stations"]}</b> (one per raceway)</td></tr>'
                f'<tr><td>Thermal cameras (both sides of the race)</td><td><b>{p["thermal"]}</b></td></tr>'
                f'<tr><td>RGB-depth cameras (gait and face recognition)</td><td><b>{p["rgbd"]}</b></td></tr>'
                f'<tr><td>Directional microphones (coughs)</td><td><b>{p["mics"]}</b></td></tr>'
                f'<tr><td>Edge computers (all processing on the farm)</td><td><b>{p["edge"]}</b></td></tr>'
                f'<tr><td>Soft-isolation pen capacity</td><td><b>≥ {p["pen_capacity"]}</b> animals</td></tr>'
                f'</table><p class="fw-muted" style="margin-top:.6rem">Planning estimate: one full pass takes about '
                f'<b>{p["minutes"]:.0f} min</b> at ~7 s per sheep. '
                f'{"This is within the ~45 min planning target." if ok else f"Above the ~45 min target: consider {p["recommended"]} raceways."} '
                f'These are planning assumptions, not supplier specifications.</p></div>', unsafe_allow_html=True)
    elif cur == 1:
        st.markdown("#### Starting without a history")
        st.markdown('<p class="fw-muted">The system compares every animal with its own normal, so it needs a few weeks of readings '
                    'before it can judge anyone. Temporary collars on a small group are intended to speed this up '
                    'without buying permanent collars for everyone.</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            s.cfg["collars"] = st.toggle("Use temporary sentinel collars", s.cfg["collars"])
            s.cfg["pct"] = st.slider("Share of the flock wearing a collar", 5, 10, s.cfg["pct"], format="%d%%",
                                     disabled=not s.cfg["collars"])
            s.cfg["weeks"] = st.slider("Baseline period (weeks)", 2, 3, s.cfg["weeks"])
        with c2:
            k = math.ceil(s.cfg["n"] * s.cfg["pct"] / 100)
            if s.cfg["collars"]:
                st.markdown(
                    f'<div class="fw-card mint"><h4>{k} collars for {s.cfg["weeks"]} weeks</h4>'
                    f'<p class="fw-muted">Instead of {s.cfg["n"]} permanent collars. After the baseline period the collars come off '
                    f'and the raceway station takes over.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fw-card"><h4>Station only</h4><p class="fw-muted">Baselines build from raceway passages alone. '
                            'Early alerts will be less reliable for the first weeks.</p></div>', unsafe_allow_html=True)
        st.caption("Demo note: collar data are not simulated. The demo simulates the baseline period itself "
                   f"({s.cfg['weeks'] * 7} days of healthy passages) before monitoring starts.")
    elif cur == 2:
        st.markdown("#### What is required and what is optional")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="fw-card"><h4>Required: the passage station</h4>'
                '<span class="fw-chip ok">Thermal camera</span><span class="fw-chip ok">RGB-depth camera</span>'
                '<span class="fw-chip ok">Directional microphone</span>'
                '<p class="fw-muted" style="margin-top:.5rem">Reads eye-region temperature, gait, coughing, walking speed and visible skin lesions '
                'as each sheep walks through. Sheep are recognised by their faces: no ear tags to scan, no wearables to carry.</p></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                '<div class="fw-card mint"><h4>Optional: quiet electric sorting gate</h4>'
                '<p class="fw-muted">An electric servo-motor gate that diverts a flagged animal into a mesh isolation pen where it can still see, '
                'hear and smell the flock. Designed to be quieter than pneumatic isolators. You can pause it at any moment with the stop switch, '
                'and every diversion is reversible.</p></div>', unsafe_allow_html=True)
            s.cfg["servo"] = st.toggle("Install the servo gate (automatic diversion)", s.cfg["servo"])
            if not s.cfg["servo"]:
                st.caption("Without the gate the system only asks you to inspect flagged animals; no gate action is taken.")
    else:
        st.markdown("#### How your data is handled")
        st.markdown(
            '<div class="fw-card"><ul class="fw-muted" style="margin:0;padding-left:1.1rem">'
            '<li><b>Raw video and audio stay on the farm.</b> They are processed locally on encrypted edge hardware; only readings and scores are kept.</li>'
            '<li><b>The farm owns its data.</b> It cannot be sold or used for commercial purposes.</li>'
            '<li><b>Regional summaries are coarse.</b> Anything shared for regional monitoring contains pattern counts per 25 km grid cell, '
            'with no animal IDs and no exact location.</li>'
            '<li><b>One exception is legal, not optional.</b> If you notify the authority of a suspected notifiable disease, that report identifies your holding. '
            'You make that notification; the system never sends it.</li></ul></div>', unsafe_allow_html=True)
        ok = st.checkbox("I have read how my data is handled", value=False, key="priv_ok")
        st.write("")
        if st.button("Start monitoring", type="primary", disabled=not ok):
            start_monitoring()
            st.rerun()
        st.caption("Design commitments of the project. This demo stores nothing outside this browser session.")

    st.write("")
    b1, b2, _ = st.columns([1, 1, 6])
    if cur > 0 and b1.button("Back"):
        s.wstep -= 1
        st.rerun()
    if cur < 3 and b2.button("Next", type="primary"):
        s.wstep += 1
        st.rerun()


def start_monitoring():
    s = st.session_state
    c = s.cfg
    ctx = core.FarmContext(system_type=c["system"], gate_mode=core.GATE_AUTO if c["servo"] else core.GATE_INSPECT,
                           audit_fraction=0.03)
    arms = (0.99, 0.995, 0.998) if c["n"] >= 60 else None   # stricter alert quantiles for larger flocks
    with st.spinner(f"Building baselines from {c['weeks'] * 7} days of healthy passages…"):
        s.fw = core.FlockWatch(seed=s.seed, context=ctx, n_animals=c["n"], daily_events=0.3,
                               warmup_rounds=c["weeks"] * 7, sentinel_pct=c["pct"], arms=arms)
    s.start_date = dt.date.today()
    s.configured = True
    s.sel = None
    s.watch_notes = []
    s.show_all_watch = False
    advance(1)


# ---------------------------------------------------------------------------
# 4. Sidebar demo controls
# ---------------------------------------------------------------------------

def sidebar():
    fw = fw_()
    with st.sidebar:
        st.markdown("### Demo controls")
        st.caption("Presentation helpers. They are not part of the farmer product.")
        if st.button("Next day", width="stretch", type="primary"):
            with st.spinner("Simulating a passage…"):
                advance(1)
            st.rerun()
        if st.button("Skip 7 days", width="stretch"):
            with st.spinner("Simulating a week…"):
                advance(7)
            st.rerun()
        if st.button("Introduce a simulated pox-like case", width="stretch"):
            aid = inject_case()
            st.session_state.flash = (f"Simulated case introduced in {aid}." if st.session_state.reveal
                                      else "A simulated case has been introduced. Advance days to see whether the station finds it.")
            st.rerun()
        if st.button("Simulate vet assessment of open alerts", width="stretch",
                     help="Uses the simulated ground truth. Reportable cases are notified first, as in the real sequence."):
            log = fw.auto_resolve_pending()
            st.session_state.flash = "; ".join(log) if log else "No open alerts."
            st.rerun()
        st.session_state.auto_audit = st.toggle("Weekly random check (3% of unflagged)", st.session_state.auto_audit)
        if st.button("Run a random check now", width="stretch"):
            res = fw.run_audit()
            missed = [r for r in res if r["result"] != "healthy"]
            st.session_state.flash = (f"Random check: {len(res)} animals examined, {len(missed)} missed case(s) found."
                                      if res else "No eligible animals.")
            st.rerun()
        st.session_state.reveal = st.toggle("Reveal simulated ground truth", st.session_state.reveal)
        with st.expander("Legal-suspicion rule (placeholder)"):
            st.caption("Where an operational alert becomes a legal suspicion must be agreed with veterinary authorities. "
                       "This is configurable here only to show the consequence.")
            idx = core.SUSPICION_RULES.index(fw.context.suspicion_rule)
            fw.context.suspicion_rule = st.selectbox("Rule", core.SUSPICION_RULES, index=idx,
                                                     help="Applies to alerts raised from now on.")
        st.divider()
        if st.button("Back to setup", width="stretch"):
            for k in ("fw", "configured", "sel"):
                st.session_state.pop(k, None)
            st.session_state.wstep = 0
            st.rerun()


# ---------------------------------------------------------------------------
# 5. Monitoring tab
# ---------------------------------------------------------------------------

def header(fw, states):
    s = st.session_state
    counts = {k: 0 for k in ("normal", "watch", "inspect", "isolate", "report", "official")}
    for st_, _ in states.values():
        counts[st_] += 1
    last = [f for f in fw.feed if f["round"] == fw.round_no]
    if last:
        t0, t1 = min(f["pass_s"] for f in last), max(f["pass_s"] for f in last)
        passage = f"Passage today {when(fw.round_no, t0)[-5:]} to {when(fw.round_no, t1)[-5:]}"
    else:
        passage = "No passage today (handling round)"
    warns = [w for w in fw.warnings if w["round"] == fw.round_no]
    if any(w["type"] == "not observed" for w in warns):
        health = '<span class="fw-chip alert">Some animals not observed: manual check needed</span>'
    elif warns:
        health = f'<span class="fw-chip warn">System OK: {len(warns)} data notice(s) today</span>'
    else:
        health = '<span class="fw-chip ok">System healthy</span>'
    if fw.context.gate_mode == core.GATE_INSPECT:
        gate = '<span class="fw-chip neutral">Gate: inspection requests only</span>'
    elif fw.context.gate_paused:
        gate = '<span class="fw-chip warn">Gate paused by you</span>'
    else:
        gate = '<span class="fw-chip ok">Gate active</span>'
    day = when(fw.round_no)[:-7]
    st.markdown(
        f'<div class="fw-hero"><div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap">'
        f'<div><h1>{esc(s.cfg["farm_name"])}</h1><p class="fw-muted" style="margin:.2rem 0 .6rem 0">{day} · day {fw.round_no} of monitoring · '
        f'{s.cfg["n"]} sheep · {esc(fw.context.system_type.split(" (")[0])}</p>'
        f'<span class="fw-chip neutral">{passage}</span>{health}{gate}</div></div></div>', unsafe_allow_html=True)
    if fw.context.gate_mode == core.GATE_AUTO:
        fw.context.gate_paused = st.toggle("Pause the gate (your stop switch: flagged animals are then only reported, not diverted)",
                                           value=fw.context.gate_paused, key="gate_pause")
    return counts, warns


def stat(label, n, cls, sub=""):
    return f'<div class="fw-stat {cls}"><div class="n">{n}</div><div class="l">{label}</div><div class="l">{sub}</div></div>'


def flock_map(fw, states):
    pens = {}
    for aid, a in fw.animals.items():
        pens.setdefault(a.pen, []).append(aid)
    blocks = []
    for pen in sorted(pens):
        dots = "".join(f'<span class="dot {states[a][0] if states[a][0] != "normal" else ""}" '
                       f'title="{a} · {fw.animals[a].breed} · {TIER_LABEL.get(states[a][0], "Normal")}"></span>' for a in pens[pen])
        blocks.append(f'<div class="fw-pen"><div class="pl">Pen {pen}</div><div class="fw-dots">{dots}</div></div>')
    legend = ('<div class="fw-legend"><span><i class="dot"></i>Normal, passing routinely</span>'
              '<span><i class="dot watch"></i>Watchlist</span><span><i class="dot isolate"></i>Soft isolation</span>'
              '<span><i class="dot inspect"></i>Inspection requested</span><span><i class="dot report"></i>Report suspicion</span></div>')
    st.markdown(f'<div class="fw-card"><h4>The flock at a glance</h4><div style="display:flex;flex-wrap:wrap">{"".join(blocks)}</div>{legend}</div>',
                unsafe_allow_html=True)


def animal_card(fw, aid, state, rec, selected, compact=False):
    a = fw.animals[aid]
    sel_cls = "sel_" if selected else ""
    with st.container(key=f"card_{sel_cls}{state}_{aid}"):
        if rec is not None:
            ts = when(rec.round_no, rec.pass_s)
            if rec.source == "audit":
                reason = "Found by the weekly random check, not by the station."
            else:
                reason = short_signal(fw, aid, rec.z, rec.resid)
            action = {"report": "Suspected notifiable disease: your decision to notify.",
                      "isolate": f"Diverted to the soft-isolation pen {'(gate paused earlier)' if False else ''}".strip(),
                      "inspect": "Please inspect it. The gate took no action."}[state]
        else:
            rd = fw.readings[aid][-1]
            ts = when(rd["round"], rd["pass_s"])
            reason = short_signal(fw, aid, rd["z"], adjusted_values(fw, rd["raw"], rd["ambient"])) if rd["z"] else ""
            action = "Re-checked at the next passage."
        st.markdown(
            f'<span class="fw-tier {state}">{TIER_LABEL[state]}</span><div class="fw-id">{aid}</div>'
            f'<div class="fw-muted">{a.breed} · pen {a.pen} · {ts}</div>'
            f'<p style="margin:.55rem 0 .15rem 0">{esc(reason)}</p><div class="fw-muted">{esc(action)}</div>', unsafe_allow_html=True)
        if rec is not None:
            st.button("Review & provide feedback", key=f"rev_{rec.id}", type="primary", on_click=select, args=("alert", rec.id))
        else:
            st.button("Review", key=f"revw_{aid}", on_click=select, args=("animal", aid))


def monitoring_tab():
    s = st.session_state
    fw = fw_()
    states = compute_states(fw)
    counts, warns = header(fw, states)
    n = len(fw.animals)

    cols = st.columns(4)
    cols[0].markdown(stat(f"animals normal and passing routinely", counts["normal"], "green", f"of {n} sheep"), unsafe_allow_html=True)
    cols[1].markdown(stat("on the watchlist", counts["watch"], "amber", "re-checked next passage"), unsafe_allow_html=True)
    cols[2].markdown(stat("soft isolation or inspection", counts["isolate"] + counts["inspect"], "amber", "awaiting your feedback"), unsafe_allow_html=True)
    cols[3].markdown(stat("suspected notifiable disease", counts["report"], "red", "you decide whether to notify"), unsafe_allow_html=True)
    st.write("")

    sel = s.get("sel")
    if sel and sel[0] == "animal" and states[sel[1]][1] is not None:
        sel = ("alert", states[sel[1]][1].id)
        s.sel = sel

    attention = [(a, st_, r) for a, (st_, r) in states.items() if st_ in ("report", "isolate", "inspect")]
    attention.sort(key=lambda t: ({"report": 0, "isolate": 1, "inspect": 1}[t[1]], t[2].round_no * -1))
    watch = [(a, st_, r) for a, (st_, r) in states.items() if st_ == "watch"]
    watch.sort(key=lambda t: -(fw.readings[t[0]][-1]["S"] or 0))

    def is_sel(kind, key):
        return sel is not None and sel[0] == kind and sel[1] == key

    left, right = (st.columns([4, 7], gap="large") if sel else (st.container(), None))
    with left:
        st.markdown(f"#### Needs your attention ({len(attention)})")
        if not attention:
            st.markdown('<div class="fw-card mint">Nothing needs a closer look right now. The station keeps checking every passage.</div>',
                        unsafe_allow_html=True)
        grid = st.columns(1 if sel else 2)
        for i, (aid, st_, rec) in enumerate(attention):
            with grid[i % len(grid)]:
                animal_card(fw, aid, st_, rec, is_sel("alert", rec.id))
        st.markdown(f"#### Watchlist ({len(watch)})")
        st.caption("A mild deviation. It never triggers isolation on its own. Persistent deviation (2 of the last 3 passes) moves an animal up.")
        shown = watch if s.get("show_all_watch") else watch[:6]
        grid = st.columns(1 if sel else 3)
        for i, (aid, st_, rec) in enumerate(shown):
            with grid[i % len(grid)]:
                animal_card(fw, aid, st_, rec, is_sel("animal", aid), compact=True)
        if len(watch) > 6:
            st.toggle(f"Show all {len(watch)} watchlist animals", key="show_all_watch")
    if sel:
        with right:
            drawer(fw, states, sel)

    st.write("")
    flock_map(fw, states)
    st.write("")
    with st.expander("Passage feed: latest animals recognised by face (no tags, no wearables)"):
        rows = [f for f in reversed(fw.feed[-8:])]
        if not rows:
            st.caption("No passages yet.")
        else:
            body = "".join(
                f'<tr><td>{when(f["round"], f["pass_s"])}</td><td><b>{f["animal"]}</b> · {fw.animals[f["animal"]].breed}</td>'
                f'<td>{f["id_conf"]:.0%}</td><td>{ {0: "Normal", 1: "Watchlist", 2: "Alert"}.get(f["tier"], "Building baseline")}</td></tr>'
                for f in rows)
            st.markdown(f'<table class="fw-table"><tr><th>When</th><th>Animal</th><th>Face-match confidence</th><th>Result</th></tr>{body}</table>',
                        unsafe_allow_html=True)
        if warns:
            st.markdown("**Data notices today**")
            for w in warns:
                st.caption(f"{w['animal']} · {w['type']}: {w['detail']}")
        st.caption("Readings with low face-match confidence are not linked to an animal's history and not used for learning.")


# ----- drawer ----------------------------------------------------------------

def telemetry_table(fw, aid):
    rows = fw.readings[aid]
    if not rows:
        st.caption("No readings yet for this animal.")
        return
    start = fw.start_mean.get(aid, {})
    last3 = rows[-3:]
    head = '<th>Indicator</th><th>Average at start</th>' + "".join(
        f'<th>{"Latest" if i == len(last3) - 1 else f"{len(last3) - 1 - i} passage{"s" if len(last3) - 1 - i > 1 else ""} earlier"}</th>' for i in range(len(last3)))
    labels = {"thermal": "Thermal delta (°C)", "gait": "Gait asymmetry (/100)", "cough": "Cough count (per passage)",
              "speed": "Activity change", "lesion": "Skin lesion score (/100)"}
    body = ""
    for f in DISPLAY_FEATURES:
        s0 = start.get(f, last3[0]["raw"][f])
        cells = ""
        for rd in last3:
            v = adjusted_values(fw, rd["raw"], rd["ambient"])[f]
            if f == "thermal":
                txt = f"{v - s0:+.2f} °C"
            elif f == "speed":
                txt = f"{(v / s0 - 1) * 100:+.0f}%"
            elif f == "cough":
                txt = f"{v:.1f}"
            else:
                txt = f"{v:.0f}"
            flag = (rd["z"] is not None and abs(rd["z"][f]) >= 2.0
                    and not (f in ("gait", "cough", "lesion") and rd["z"][f] < 0))
            cells += f'<td class="{"flag" if flag else ""}">{txt}</td>'
        s_txt = {"thermal": f"baseline {s0:.1f} °C", "speed": f"{s0:.2f} m/s", "cough": f"{s0:.1f}"}.get(f, f"{s0:.0f}")
        body += f"<tr><td>{labels[f]}</td><td>{s_txt}</td>{cells}</tr>"
    st.markdown(f'<table class="fw-table"><tr>{head}</tr>{body}</table>', unsafe_allow_html=True)
    st.caption(f"Latest passage {when(last3[-1]['round'], last3[-1]['pass_s'])}. Shaded cells are at least 2 standard deviations from this animal's own baseline.")


def score_chart(fw, aid):
    h = fw.history.get(aid, [])[-14:]
    if len(h) < 2:
        st.caption("Not enough scored passages for a trend yet.")
        return
    df = pd.DataFrame(h)
    df["day"] = df["round"]
    base = alt.Chart(df).encode(x=alt.X("day:Q", title="Monitoring day", axis=alt.Axis(tickMinStep=1, grid=False)))
    line = base.mark_line(color=GREEN, strokeWidth=2, point=alt.OverlayMarkDef(color=GREEN, size=45)).encode(
        y=alt.Y("S:Q", title="Anomaly score", scale=alt.Scale(zero=False)))
    watch = base.mark_line(color="#F6AD55", strokeDash=[4, 3]).encode(y="theta_watch:Q")
    alert = base.mark_line(color=AMBER, strokeDash=[6, 3]).encode(y="theta_alert:Q")
    st.altair_chart((line + watch + alert).properties(height=170).configure_view(strokeWidth=0)
                    .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor="#CBD5E0"), width="stretch")
    st.caption("Green: this animal's score. Light amber dashes: watchlist threshold. Amber dashes: alert threshold.")


def ladder(state):
    def cls(name, order):
        return "on " + name if (state in order) else ""
    st.markdown(
        '<div class="fw-ladder">'
        f'<div class="{cls("watch", ("watch",))}"><b>Watchlist</b>Keep an eye on it</div>'
        f'<div class="{cls("isolate", ("isolate", "inspect"))}"><b>Soft isolation</b>Mesh pen, flock contact kept</div>'
        f'<div class="{cls("report", ("report", "official"))}"><b>Report suspicion</b>Vet and authority: your call</div></div>',
        unsafe_allow_html=True)


def drawer(fw, states, sel):
    s = st.session_state
    kind, key = sel
    rec = fw.get_alert(key) if kind == "alert" else None
    aid = rec.animal_id if rec else key
    a = fw.animals[aid]
    state = states[aid][0] if rec is None or not rec.resolved else ("official" if a.status == "official" else "normal")
    if rec is not None and not rec.resolved:
        state = states[aid][0]

    with st.container(key="drawer"):
        c1, c2 = st.columns([5, 2])
        c1.markdown(f'<div class="fw-id" style="font-size:1.5rem">{aid}</div><div class="fw-muted">{a.breed} · pen {a.pen}'
                    f'{" · sentinel cohort" if a.sentinel and s.cfg["collars"] else ""}</div>', unsafe_allow_html=True)
        c2.button("✕ Close", key="close_drawer", on_click=close_drawer)
        if rec is not None and rec.resolved:
            st.markdown(f'<span class="fw-chip ok">Resolved: {esc(rec.outcome)} ({esc(rec.evidence)})</span>', unsafe_allow_html=True)
        elif state in TIER_LABEL:
            st.markdown(f'<span class="fw-chip {"alert" if state == "report" else "warn"}">{TIER_LABEL[state]}</span>', unsafe_allow_html=True)
        if s.reveal and rec is not None:
            st.caption(f"Simulated ground truth: {'real disease event' if rec.truth_disease else 'no real disease (transient or noise)'}"
                       f"{' · true pattern: ' + rec.truth_pattern if rec.truth_pattern else ''}")

        st.markdown("##### Sensor readings")
        telemetry_table(fw, aid)
        score_chart(fw, aid)

        st.markdown("##### Why this was flagged")
        if rec is not None and rec.source == "audit":
            st.write("The station had not flagged this animal. The weekly random check of unflagged animals found signs of illness on clinical examination.")
        else:
            if rec is not None:
                z, vals = rec.z, rec.resid
                thr = 2.0
            else:
                rd = fw.readings[aid][-1]
                z, vals = rd["z"], adjusted_values(fw, rd["raw"], rd["ambient"])
                thr = 1.5
            ph = signal_phrases(fw, aid, z, vals, thr) if z else []
            for _, t in ph:
                st.markdown(f"- {t}")
            if not ph:
                st.markdown("- Several small deviations that are only unusual together.")
            if rec is not None:
                st.caption(f"Above the alert threshold on {core.K_PERSIST} of the last {core.N_PERSIST} passages. "
                           f"Closest pattern: {rec.pattern}. This is a statistical deviation from this animal's own baseline, not a diagnosis.")
                if rec.exposure_contacts:
                    st.caption("Recent contact with flagged animals: " + ", ".join(f"{x} ({e:.2f})" for x, e in rec.exposure_contacts[:4]))
            else:
                st.caption("Deviation from this animal's own baseline, not a diagnosis. No action is required from you.")
        if rec is not None:
            with st.expander("Technical detail for your vet"):
                st.write(fw.explain(rec))
                st.download_button("Download vet dossier (JSON)", json.dumps(fw.build_dossier(rec.id), indent=2, default=str),
                                   file_name=f"dossier_{aid}_{rec.id}.json", mime="application/json", key=f"dl_{rec.id}")

        st.markdown("##### Recommended action")
        ladder(state if rec is None or not rec.resolved else "normal")
        advice = {
            "watch": "No action needed. The animal is re-checked at the next passage. If you notice anything yourself, add a note.",
            "inspect": "Look at this animal today and record what you find. The gate took no action, so it is still with the flock.",
            "isolate": "The gate diverted it to the soft-isolation pen. It can still see, hear and smell the flock. Look at it today and record what you find.",
            "report": "Treat this as a suspected notifiable disease. Call your vet and notify the competent authority now: do not wait for a pen-side test. "
                      "Keep the animal separated and follow the instructions you receive. You make the notification; this system never does.",
            "official": "This case is with the official pathway. It is outside this system's scope.",
            "normal": "",
        }[state]
        if advice:
            st.markdown(f'<div class="fw-card {"" if state != "report" else ""}">{advice}</div>', unsafe_allow_html=True)

        if rec is None:
            note = st.text_input("Add a note about this animal", key=f"wn_{aid}", placeholder="e.g. slightly dull this morning")
            if st.button("Save note", key=f"wns_{aid}") and note.strip():
                s.watch_notes.append({"day": fw.round_no, "animal": aid, "note": note.strip()})
                s.flash = "Note saved."
                st.rerun()
            st.caption("Notes on watchlist animals are kept in your history. They are not used for learning.")
            return
        feedback_form(fw, rec)


def feedback_form(fw, rec):
    s = st.session_state
    st.markdown("##### Your feedback")
    if rec.resolved:
        st.markdown(f'<div class="fw-card mint"><b>{esc(rec.outcome)}</b><br>Evidence: {esc(rec.evidence)}'
                    f'{"<br>Notes: " + esc(rec.notes) if rec.notes else ""}</div>', unsafe_allow_html=True)
        return

    # --- escalation steps (human actions only) ---
    c1, c2 = st.columns(2)
    with c1:
        if rec.vet_contacted_round is None:
            if st.button("🩺 I have contacted my vet", key=f"vet_{rec.id}"):
                fw.notify(rec.id, vet=True)
                s.flash = "Vet contact recorded."
                st.rerun()
        else:
            st.markdown(f'<span class="fw-chip ok">🩺 Vet contacted {when(rec.vet_contacted_round)[:-7]}</span>', unsafe_allow_html=True)
    with c2:
        if rec.authority_notified_round is None:
            if st.button("🏛 I have notified the authority", key=f"auth_{rec.id}", type="primary" if rec.suspicion else "secondary"):
                fw.notify(rec.id, authority=True)
                s.flash = "Notification recorded. The system did not send it: you did."
                st.rerun()
        else:
            st.markdown(f'<span class="fw-chip ok">🏛 Authority notified {when(rec.authority_notified_round)[:-7]}</span>', unsafe_allow_html=True)
    if rec.suspicion and rec.authority_notified_round is None:
        st.caption("A suspected notifiable disease is reported on suspicion. A negative pen-side test or your own inspection cannot clear it.")

    # --- pen-side test ---
    if rec.pattern in PENTEST_PATTERNS:
        with st.container():
            r1, r2 = st.columns([3, 1])
            res = r1.radio("Pen-side rapid test (RPA strip)", ["Not done", "Positive", "Negative"], horizontal=True, key=f"pt_{rec.id}",
                           index=0 if rec.pen_test is None else (1 if rec.pen_test == "positive" else 2))
            if r2.button("Record", key=f"ptr_{rec.id}") and res != "Not done":
                fw.record_pen_test(rec.id, res.lower())
                s.flash = f"Pen-side test recorded as {res.lower()} (presumptive)."
                st.rerun()
        st.caption("Presumptive only. A positive result strengthens the suspicion; a negative result does not clear it. "
                   "Confirmation is by the reference laboratory.")
        if rec.pen_test:
            st.markdown(f'<span class="fw-chip {"alert" if rec.pen_test == "positive" else "neutral"}">🧪 Pen-side test {rec.pen_test} (presumptive)</span>',
                        unsafe_allow_html=True)

    # --- outcome ---
    with st.form(key=f"outcome_{rec.id}"):
        choice = st.selectbox("What did you or your vet find?", list(OUTCOME_CHOICES), key=f"oc_{rec.id}")
        notes = st.text_area("Notes and observations", key=f"notes_{rec.id}", height=80)
        submitted = st.form_submit_button("Save outcome", type="primary")
    if submitted:
        outcome, evidence = OUTCOME_CHOICES[choice]
        try:
            fw.record_outcome(rec.id, outcome, evidence, notes)
            s.flash = ("Outcome saved. It is kept open for follow-up." if outcome == core.OUT_UNC
                       else "Outcome saved. Thank you: the system weighs it by how strong the evidence is.")
            st.rerun()
        except core.SequenceViolation as e:
            st.error(str(e))


# ---------------------------------------------------------------------------
# 6. History and performance tab
# ---------------------------------------------------------------------------

def reliability(n, need=20):
    return "Too few cases to judge" if n < need else "Indicative"


def history_tab():
    fw = fw_()
    m = fw.metrics()
    st.markdown("#### How reliable is the system?")
    st.markdown('<p class="fw-muted">All numbers come from simulated passages and simulated outcomes. '
                'On a real farm they would come from your recorded feedback and veterinary results.</p>', unsafe_allow_html=True)

    def pct(x):
        return "no data yet" if x is None else f"{x:.0%}"

    aud = m["audit"]
    audit_val = (f"{aud['pos']} missed case(s) in {aud['n']} animals checked" if aud["n"] else "no checks yet")
    rows = [
        ("Disease identification accuracy", pct(m["pattern_match_rate"]),
         "Confirmed cases where the closest pattern matched the true condition.", f"n = {m['pattern_match_n']}: {reliability(m['pattern_match_n'])}"),
        ("False alarm rate", pct(m["false_alarm_rate"]),
         "Resolved alerts that turned out not to be a health problem.", f"n = {m['resolved_alerts']}: {reliability(m['resolved_alerts'])}"),
        ("Random check: missed cases", audit_val,
         "Ill animals found among animals the station had NOT flagged. This is a miss rate among unflagged animals, not sensitivity.",
         aud["text"]),
        ("Alerts per day", f"{m['alerts_per_round']}", "Station alerts (excluding random-check finds).", f"{m['rounds']} days observed"),
        ("Average days from onset to alert", "no data yet" if m["mean_detection_latency_rounds"] is None else f"{m['mean_detection_latency_rounds']}",
         "Only known in simulation.", ""),
        ("Unnecessary separations", f"{m['unnecessary_separations']} ({m['unnecessary_isolation_rounds']} isolation-days)",
         "Animals diverted that turned out healthy. A welfare cost, tracked on purpose.", ""),
        ("Suspicions awaiting notification", f"{m['suspicions_not_yet_notified']}",
         "Suspected notifiable cases where nobody has recorded a notification yet.", ""),
    ]
    body = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td class='fw-muted'>{c}</td><td class='fw-muted'>{d}</td></tr>" for a, b, c, d in rows)
    st.markdown(f'<div class="fw-card"><table class="fw-table"><tr><th>Measure</th><th>Value</th><th>What it means</th><th>Confidence</th></tr>{body}</table></div>',
                unsafe_allow_html=True)
    st.caption("The 3% weekly check only bounds the miss rate among unflagged animals once roughly 100 of them have been examined "
               "(rule of three: 3 / n). Early on the bound is very wide.")

    st.markdown("#### Feedback history")
    recs = list(reversed(fw.alerts))
    if not recs:
        st.markdown('<div class="fw-card mint">No alerts yet. Skip a few days from the demo controls.</div>', unsafe_allow_html=True)
    else:
        data = []
        for r in recs:
            a = fw.animals[r.animal_id]
            tier = ("Report suspicion" if r.suspicion else ("Soft isolation" if r.gate_action in ("diverted", "found by audit") else "Inspection requested"))
            data.append({
                "When": when(r.round_no, r.pass_s), "Animal": r.animal_id, "Breed": a.breed, "Pattern": r.pattern, "Tier": tier,
                "Found by": "🎯 Weekly random check" if r.source == "audit" else "📷 Raceway station",
                "Evidence": EVIDENCE_ICON.get(r.evidence, "⏳ Awaiting feedback"),
                "Pen-side test": {"positive": "🧪 Positive", "negative": "🧪 Negative"}.get(r.pen_test, "—"),
                "Vet contacted": "🩺 " + when(r.vet_contacted_round)[:-7] if r.vet_contacted_round is not None else "—",
                "Authority notified": "🏛 " + when(r.authority_notified_round)[:-7] if r.authority_notified_round is not None else "—",
                "Outcome": r.outcome or "Open", "Notes": r.notes,
            })
        df = pd.DataFrame(data)
        f1, f2 = st.columns(2)
        ev = f1.multiselect("Filter by evidence", sorted(df["Evidence"].unique()))
        oc = f2.multiselect("Filter by outcome", sorted(df["Outcome"].unique()))
        if ev:
            df = df[df["Evidence"].isin(ev)]
        if oc:
            df = df[df["Outcome"].isin(oc)]
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption("Click a column header to sort. 🩺 veterinary assessment · 🏛 reference laboratory or official notification · 🧪 pen-side test (presumptive) · 👁 farmer observation · 🎯 random check.")
    if st.session_state.watch_notes:
        with st.expander("Watchlist notes"):
            st.dataframe(pd.DataFrame(st.session_state.watch_notes), width="stretch", hide_index=True)

    with st.expander("How the system adapts its alert threshold (advanced)"):
        st.write("A simplified, conservative contextual bandit chooses one alert quantile for the whole flock each day. It only departs from the "
                 "baseline quantile when its cumulative reward stays above a safety margin. Labels count in proportion to the strength of their evidence "
                 "(farmer 0.4, pen-side test 0.7, vet or laboratory 1.0). Uncertain outcomes and sensor faults never train it.")
        st.dataframe(pd.DataFrame(fw.bandit.table()), width="stretch", hide_index=True)
        st.caption(f"Reward table (a value judgement to agree with vets and farmers): {core.REWARDS}. With these numbers, alerting pays off when the "
                   f"estimated illness probability exceeds {core.implied_alert_probability():.2f}. This is NOT the algorithm of Kazerouni et al. (2017) "
                   "and carries no formal guarantee.")


# ---------------------------------------------------------------------------
# 7. Welfare and values tab
# ---------------------------------------------------------------------------

def welfare_tab():
    fw = fw_()
    m = fw.metrics()
    states = compute_states(fw)
    attention = sum(1 for v in states.values() if v[0] in ("report", "isolate", "inspect", "watch"))
    n = len(fw.animals)
    st.markdown("#### Why this benefits you and your animals")
    st.markdown('<p class="fw-muted">These are the intended benefits of the design. They still have to be shown on real farms; '
                'the demo only makes them visible.</p>', unsafe_allow_html=True)

    st.markdown("##### For the animals")
    cards = [
        ("Quiet, predictable handling", "An electric servo gate is designed to be quieter than a pneumatic isolator, so a flagged animal is moved with less panic and noise."),
        ("Company during isolation", "The mesh pen lets a separated animal keep seeing, hearing and smelling the flock, which limits the stress of social isolation."),
        ("Help earlier", "Gradual changes in temperature, gait and coughing can show up before they are obvious by eye, so pain and illness may be noticed sooner."),
        ("Nothing to wear", "No ear tags to scan and no collars on the flock. Sheep are recognised by their faces; only a small cohort wears temporary collars during the baseline weeks."),
        ("Compared with itself", "Each sheep is judged against its own normal, not a breed average, so a naturally warm or slow animal is not treated as a problem."),
        ("Fewer needless separations", f"Every diversion that turns out unnecessary is counted as a welfare cost. So far: {m['unnecessary_separations']} separation(s), {m['unnecessary_isolation_rounds']} isolation-day(s)."),
    ]
    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (t, d) in zip(cols, cards[i:i + 3]):
            col.markdown(f'<div class="fw-card tall"><h4>{t}</h4><p class="fw-muted" style="margin:0">{d}</p></div>', unsafe_allow_html=True)
        st.write("")

    st.markdown("##### For you")
    cards = [
        ("You stay in control", "The system advises and never decides. It cannot cull, cannot notify the authority, and its gate has a stop switch you control."),
        ("Look at fewer animals", f"Right now {attention} of {n} sheep need any attention at all. The rest pass without you checking them."),
        ("Earlier warning", "Spotting an outbreak earlier means fewer animals exposed and more options than culling. This is the aim, not yet a measured result."),
        ("A ready file for your vet", "Every alert comes with a dossier: readings, trends, contacts with flagged animals and what you did, with timestamps."),
        ("Clear on your legal duties", "Suspected notifiable disease is reported on suspicion. The system reminds you and records your notification; it never waits for a rapid test."),
        ("Honest about its limits", "Random checks of unflagged animals show how many cases the station misses, so you are not asked to trust it blindly."),
    ]
    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (t, d) in zip(cols, cards[i:i + 3]):
            col.markdown(f'<div class="fw-card mint tall"><h4>{t}</h4><p class="fw-muted" style="margin:0">{d}</p></div>', unsafe_allow_html=True)
        st.write("")

    st.markdown("##### Design for values and the Five Domains")
    dfv = pd.DataFrame([
        ("Animal welfare", "Soft isolation with flock contact; quiet gate; individual baselines; unnecessary separations counted", "Health, Behavioural interactions, Physical environment"),
        ("Human agency", "Advice only; stop switch; human-only notification; negative tests cannot clear a notifiable suspicion", "Meaningful human control"),
        ("Transparency", "Plain-language reasons; technical detail and dossier on request; thresholds shown", "Explanation for farmer and vet"),
        ("Privacy and ownership", "Local processing; farm owns data; 25 km grid summaries", "Data governance"),
        ("Fairness to animals", "No breed average; low-confidence identity never used for learning; missing data is never read as healthy", "Avoiding species bias and over-datafication"),
    ], columns=["Value", "How the design supports it", "Domain or principle"])
    st.dataframe(dfv, width="stretch", hide_index=True)
    st.warning("The Five Domains include mental state, which these sensors cannot measure. Flock Watch therefore gives no overall welfare score: "
               "it flags health and behaviour changes and leaves the welfare judgement to you.")
    with st.expander("What this demo cannot show"):
        st.markdown(
            "- Every reading, event and outcome is simulated; no real detection performance is shown.\n"
            "- Collars, face recognition, microphones and cameras are not modelled individually; they are represented by simulated features.\n"
            "- The adaptive threshold is a simplified bandit without formal guarantees.\n"
            "- Where an alert becomes a legal suspicion is a placeholder to be agreed with veterinary authorities.\n"
            "- Hardware costs are not shown because no supplier quotes exist yet.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def init_state():
    s = st.session_state
    s.setdefault("configured", False)
    s.setdefault("wstep", 0)
    s.setdefault("cfg", {"farm_name": "Demo farm, Lesvos", "n": 120, "raceways": 1, "system": core.SYSTEM_INTENSIVE,
                         "collars": True, "pct": 8, "weeks": 3, "servo": True})
    s.setdefault("seed", 11)
    s.setdefault("auto_audit", True)
    s.setdefault("reveal", False)
    s.setdefault("sel", None)
    s.setdefault("watch_notes", [])
    s.setdefault("flash", None)


init_state()
if not st.session_state.configured:
    wizard()
else:
    fw = fw_()
    sidebar()
    if st.session_state.flash:
        st.toast(st.session_state.flash)
        st.session_state.flash = None
    tab1, tab2, tab3 = st.tabs(["Monitoring", "Feedback and performance", "Welfare and values"])
    with tab1:
        monitoring_tab()
    with tab2:
        history_tab()
    with tab3:
        welfare_tab()
