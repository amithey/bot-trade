"""Presentation preferences only. No trading configuration lives here."""
import json
import re
from pathlib import Path

import streamlit as st

from dashboard import theme

PRESETS = {"Sapphire": "#6788ff", "Jade": "#62b8a5", "Amethyst": "#b29ae8", "Sand": "#c5ac80"}
DEFAULTS = {"accent": PRESETS["Jade"], "surface": "Graphite"}


def _path():
    # Resolve the current owner without calling account_id(): appearance is
    # applied before the login gate, and caching ``account:unknown`` there
    # would make the newly signed-in session keep the anonymous identity.
    from dashboard import _identity
    from dashboard._shared import ROOT
    mode = _identity.auth_mode()
    if mode == "accounts" and _identity.is_logged_in():
        user = _identity.current_user()
        identity = f"user:{(user.get('email') or '').strip().lower()}"
    elif mode == "accounts":
        from dashboard import _accounts
        email = _accounts.current_email()
        identity = f"account:{email}" if email else "account:unknown"
    elif mode == "oidc":
        user = _identity.current_user()
        email = (user.get("email") or "").strip().lower()
        identity = f"user:{email}" if email else f"sub:{user.get('sub', 'unknown')}"
    elif mode == "password":
        identity = "shared"
    else:
        identity = "local"
    return ROOT / "data" / "profiles" / (
        f"{_identity.account_slug(identity)}.appearance.json"
    )


def preferences():
    path = _path()
    key = "_appearance:" + str(path)
    if key not in st.session_state:
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            values = {}
        if not isinstance(values, dict):
            values = {}
        accent = values.get("accent", DEFAULTS["accent"])
        if not isinstance(accent, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", accent):
            accent = DEFAULTS["accent"]
        st.session_state[key] = {"accent": accent, "surface": values.get("surface") if values.get("surface") in ("Graphite", "Midnight") else "Graphite"}
    return st.session_state[key]


def apply_appearance():
    values = preferences()
    accent = values["accent"]
    channels = [int(accent[i:i+2], 16) for i in (1, 3, 5)]
    foreground = "#101315" if sum(c*w for c,w in zip(channels, (.2126,.7152,.0722))) > 145 else "#ffffff"
    assets = Path(__file__).parent / "assets"
    st.logo((assets / "wordmark.svg").read_text(encoding="utf-8").replace("#2962ff", accent),
            size="large", icon_image=(assets / "mark.svg").read_text(encoding="utf-8").replace("#2962ff", accent))
    rgb = ",".join(str(int(accent[i:i+2], 16)) for i in (1, 3, 5))
    css = theme._THEME_CSS.replace(theme.ACCENT, accent).replace(theme.CYAN, accent)
    css = css.replace("rgba(41,98,255,", f"rgba({rgb},")
    if values["surface"] == "Graphite":
        css = css.replace(theme.BG_DEEP, "#111315").replace(theme.BG_PANEL, "#1a1d20").replace(theme.BORDER, "#2b3034")
    controls = (
        f"<style>:root{{--bt-accent:{accent};}} "
        "[data-testid='stBaseButton-primary'],"
        "[data-testid='stBaseButton-primaryFormSubmit'],"
        "[data-testid='stBaseButton-segmented_controlActive']"
        f"{{background:{accent}!important;color:{foreground}!important;"
        f"border-color:{accent}!important;}}</style>"
    )
    # One style payload gives the browser a single visual state per rerun.
    st.html(css + _MINIMAL_CSS + controls)


def appearance_settings():
    values = preferences()
    st.subheader("Make the workspace yours")
    st.caption("Choose your accent and background. Buy/sell colors stay consistent so market signals remain clear.")
    with st.form("appearance_form"):
        preset = st.selectbox("Accent preset", ["Custom", *PRESETS], index=0)
        custom = st.color_picker("Custom accent", values["accent"])
        surface = st.radio("Background", ["Graphite", "Midnight"],
                           index=0 if values["surface"] == "Graphite" else 1, horizontal=True)
        if st.form_submit_button("Save appearance", type="primary"):
            selected = {"accent": PRESETS.get(preset, custom), "surface": surface}
            path = _path()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(selected), encoding="utf-8")
                temporary.replace(path)
            except OSError:
                st.error("Could not save appearance. Please try again.")
            else:
                st.session_state["_appearance:" + str(path)] = selected
                st.rerun()


def fullscreen_control():
    """A user gesture requests real browser fullscreen, with an explicit fallback."""
    st.iframe("""<style>body{margin:0}button{width:100%;height:38px;background:transparent;border:1px solid #454b50;border-radius:7px;color:#e4e7ea;font:13px 'Segoe UI',sans-serif;cursor:pointer}button:hover{background:#282d31}button:focus-visible{outline:2px solid #62b8a5}#hint{color:#a7afb5;font:11px 'Segoe UI',sans-serif}</style>
<button id="full" title="Expand the entire workspace">⛶ Fullscreen</button><div id="hint" role="status"></div>
<script>
const doc=window.parent.document, button=document.getElementById('full');
const update=()=>button.textContent=doc.fullscreenElement?'⛶ Exit fullscreen':'⛶ Fullscreen';
button.onclick=async()=>{try{if(doc.fullscreenElement){await doc.exitFullscreen();}else{await doc.documentElement.requestFullscreen();}update();}catch(e){document.getElementById('hint').textContent='Use F11 for browser fullscreen.';}};
doc.addEventListener('fullscreenchange',update);update();
window.addEventListener('unload',()=>doc.removeEventListener('fullscreenchange',update));
</script>""", height=58)


_MINIMAL_CSS = """<style>
.block-container{padding:1.15rem 1.8rem 1.8rem!important;max-width:1660px!important}
header[data-testid="stHeader"]{height:0!important;background:transparent!important}
[data-testid="stSidebarHeader"]{padding-top:1rem;padding-bottom:.35rem}
[data-testid="stSidebarUserContent"]{padding-top:.3rem}
[data-testid="stPopoverBody"]:has([data-testid="stSelectbox"]){width:min(700px,calc(100vw - 32px))!important}
[data-testid="stSidebar"]{min-width:208px!important;max-width:208px!important}
[data-testid="stPageLink"] a{padding:.7rem .7rem;border-radius:7px;font-size:.86rem}
[data-testid="stPageLink"] a:hover{background:rgba(150,160,170,.08)}
.nav-caption{font-size:.62rem;letter-spacing:.14em;color:#818b92;margin:.5rem .6rem .9rem}
.nav-footer{font-size:.64rem;letter-spacing:.14em;color:#899198;border-top:1px solid #2b3034;margin-top:2rem;padding:.9rem .65rem}
.nav-footer span{display:block;font-size:.72rem;letter-spacing:0;margin-top:.4rem;color:#737d84}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:transparent;border:none;margin-top:1rem}
.bt-brand{padding:.4rem 0 .85rem;margin-bottom:.2rem;border-bottom:0}
.bt-brand-title{font-size:1.6rem;font-weight:500;letter-spacing:-.04em}
.bt-brand-sub{font-size:.8rem;color:#8e989f;margin-top:.35rem}
.agent-summary{display:flex;align-items:center;gap:1rem;padding:1rem 1.15rem;border:1px solid #2b3034;border-radius:10px;background:rgba(130,150,145,.035)}
.agent-orb{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;color:var(--bt-accent);background:rgba(140,170,160,.1);font-size:1.1rem;flex-shrink:0}
.agent-summary strong{font-size:.9rem;font-weight:550;color:#e4e8eb}
.agent-summary p{font-size:.78rem;color:#9da6ad;line-height:1.6;margin:.25rem 0 0}
.agent-summary .agent-state{margin-left:auto;white-space:nowrap;color:#a5aeb4;font-size:.7rem}
.kpi-strip{background:transparent;border:0;border-radius:0;padding:.15rem 0}
.kpi-item{padding:.7rem 1.15rem .7rem 0;border:0;gap:.3rem}
.kpi-label{font-size:.72rem;color:#929ca3}
.kpi-value{font-size:1.65rem;font-weight:500;letter-spacing:-.035em}
.kpi-sub{font-size:.7rem}
.workspace-header{padding:.3rem 0 1rem;margin-bottom:.4rem}
[data-testid="stMetric"]{box-shadow:none;border-radius:8px;min-height:108px}
[data-testid="stMetricValue"]{font-size:1.35rem;font-weight:500}
[data-baseweb="tab-list"]{gap:1rem}
button[data-baseweb="tab"]{font-size:.83rem;padding:.6rem .25rem}
button[data-baseweb="tab"][aria-selected="true"]{background:transparent}
.bt-section-title::before,.workspace-eyebrow::before{display:none}
.quote-bar{background:transparent;padding:.7rem .85rem!important}
@media(max-width:760px){.block-container{padding:1rem .8rem!important}.agent-summary{align-items:flex-start}.agent-summary .agent-state{display:none}.kpi-value{font-size:1.25rem}.bt-brand-title{font-size:1.35rem}.kpi-item{min-width:90px}}
</style>"""
