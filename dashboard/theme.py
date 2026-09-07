"""Trading workspace design tokens, shared CSS and Plotly theme.

Keep semantic trade colors separate from the blue interaction accent.
No account, persistence or market-data dependencies belong in this module.
"""
BG_DEEP = "#10141d"
BG_PANEL = "#191f2b"
BG_RAISED = "#252e3e"
BORDER = "#2a3343"
BORDER_HI = "#434651"
TEXT_DIM = "#a0aabc"
TEXT = "#d1d4dc"
TEXT_HI = "#f0f3fa"
ACCENT = "#2962ff"
CYAN = "#5b8cff"
C_BUY = "#26a69a"
C_SELL = "#ef5350"
C_HOLD = "#9598a1"
AMBER = "#f7a600"
GRID = "rgba(67,70,81,0.35)"
BG = BG_DEEP

_THEME_CSS = f"""
<style>
:root {{
    --font-ui:'Inter','Segoe UI',Arial,sans-serif;
    --font-mono:'Cascadia Code','SFMono-Regular',Consolas,monospace;
}}
html, body, [data-testid="stAppViewContainer"] {{
    background:{BG_DEEP}; color:{TEXT}; font-family:var(--font-ui);
    -webkit-font-smoothing:antialiased;
}}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"] {{ display:none; }}
header[data-testid="stHeader"] {{ background:{BG_DEEP}; height:2.5rem; }}
.block-container {{ max-width:100%; padding:3.2rem 1.5rem 2rem; }}
[data-testid="stVerticalBlock"] {{ gap:.85rem; }}
[data-testid="stSidebar"] {{ background:{BG_PANEL}; border-right:1px solid {BORDER}; }}
[data-testid="stSidebarNav"] a {{
    color:{TEXT_DIM}; font-size:.83rem; border-radius:8px; margin:2px 0;
}}
[data-testid="stSidebarNav"] a:hover {{ color:{TEXT_HI}; background:{BG_RAISED}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    color:{TEXT_HI}; background:rgba(41,98,255,.16); box-shadow:inset 2px 0 {ACCENT};
}}
[data-testid="stNavSectionHeader"] p {{
    font-size:.64rem; font-weight:600; letter-spacing:.09em; color:{TEXT_DIM};
    text-transform:uppercase;
}}
[data-testid="stSidebarCollapsedControl"] {{ color:{TEXT}; }}
h1,h2,h3,h4,h5 {{ font-family:var(--font-ui); color:{TEXT_HI}; letter-spacing:-.015em; }}
h3 {{ font-size:1.1rem; }}
h5 {{ font-size:.9rem; font-weight:600; }}
[data-testid="stCaptionContainer"] {{ color:{TEXT_DIM}; font-size:.76rem; }}
label, [data-testid="stWidgetLabel"] p {{ font-size:.75rem; color:{TEXT_DIM}; font-weight:500; }}
[data-baseweb="select"] > div, [data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea {{
    background:{BG_PANEL}; color:{TEXT}; border-color:{BORDER_HI}; border-radius:8px;
    font-size:.82rem;
}}
[data-testid="stTextArea"] textarea {{ line-height:1.55; }}
.stButton > button, [data-testid="stLinkButton"] a {{
    min-height:38px; border-radius:8px; background:{BG_PANEL}; border:1px solid {BORDER_HI};
    color:{TEXT}; font-size:.8rem; font-weight:500; white-space:nowrap; transition:background .12s ease;
}}
.stButton > button:hover {{ background:{BG_RAISED}; color:{TEXT_HI}; border-color:{TEXT_DIM}; }}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background:{ACCENT}; color:white; border-color:{ACCENT};
}}
button:focus-visible, a:focus-visible {{ outline:2px solid {CYAN}; outline-offset:2px; }}
button:disabled {{ opacity:.45; }}
[data-testid="stMetric"] {{
    background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:8px; padding:1rem 1.05rem;
    font-variant-numeric:tabular-nums;
}}
[data-testid="stMetricLabel"] p {{ font-size:.72rem; color:{TEXT_DIM}; }}
[data-testid="stMetricValue"] {{ font-size:1.45rem; font-weight:600; letter-spacing:-.02em; }}
[data-testid="stMetricDelta"] {{ font-size:.71rem; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border:1px solid {BORDER}; border-radius:8px; }}
[data-testid="stPlotlyChart"] {{ border:1px solid {BORDER}; border-radius:8px; overflow:hidden; }}
[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:8px; background:{BG_PANEL}; }}
[data-testid="stExpander"] summary {{ font-size:.82rem; }}
[data-testid="stAlertContainer"] {{ background:transparent; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background:rgba(247,166,0,.07); border-left:3px solid {AMBER}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ background:rgba(239,83,80,.07); border-left:3px solid {C_SELL}; }}
[data-testid="stAlert"] {{ border:1px solid {BORDER}; background:{BG_PANEL}; border-radius:8px; }}
button[data-baseweb="tab"] {{
    color:{TEXT_DIM}; padding:.65rem .8rem; font-size:.78rem; background:transparent;
}}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{TEXT_HI}; }}
[data-baseweb="tab-highlight"] {{ background:{ACCENT}; height:2px; }}
[data-baseweb="tab-list"] {{ gap:0; border-bottom:1px solid {BORDER}; }}
hr {{ border-color:{BORDER}; margin:.6rem 0; }}

/* Shared surfaces: panels hold content; blue is reserved for interaction. */
[data-testid="stVerticalBlockBorderWrapper"] > div {{ border-color:{BORDER} !important; border-radius:10px !important; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ background:{BG_PANEL}; }}
[data-testid="stSidebarNav"] {{ padding-top:.6rem; }}
[data-testid="stSidebarNav"] a {{ padding:.3rem .65rem; min-height:36px; box-sizing:border-box; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ font-weight:600; }}
[data-testid="stMetric"] {{ box-shadow:0 3px 12px rgba(0,0,0,.09); min-height:120px; }}
[data-testid="stMetricLabel"] p, .kpi-label {{ letter-spacing:.025em; }}
[data-testid="stMetricValue"], .kpi-value, .watch-price {{ font-family:var(--font-ui); font-variant-numeric:tabular-nums; }}
[data-baseweb="tab-list"] {{ overflow-x:auto; scrollbar-width:thin; gap:.25rem; }}
button[data-baseweb="tab"] {{ white-space:nowrap; border-radius:6px 6px 0 0; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background:rgba(41,98,255,.09); }}
[data-testid="stExpander"] summary {{ padding:.8rem 1rem; }}
.workspace-context {{ display:flex; align-items:center; gap:.5rem; color:{TEXT_DIM}; font-size:.72rem; white-space:nowrap; }}
.workspace-context i {{ width:6px; height:6px; border-radius:50%; background:{CYAN}; }}
.workspace-eyebrow {{ display:flex; align-items:center; gap:.5rem; }}
.workspace-eyebrow::before {{ content:""; width:16px; height:2px; background:{CYAN}; }}
.bt-section-title {{ display:flex; align-items:center; gap:.5rem; }}
.bt-section-title::before {{ content:""; width:3px; height:12px; border-radius:2px; background:{CYAN}; }}
.quote-bar {{ border-radius:8px 8px 0 0; padding:.8rem 1rem !important; }}
.empty-workspace {{ border-radius:10px; background:radial-gradient(ellipse at top,rgba(41,98,255,.06),transparent 70%) !important; }}
.research-heading {{ font-size:.78rem; color:{TEXT_HI}; font-weight:600; margin:.25rem 0 .6rem; }}
@media (max-width:760px) {{
    .workspace-context {{ display:none; }}
    [data-testid="stMarkdownContainer"] h1.page-title {{ font-size:1.4rem !important; }}
    .bt-brand-title {{ font-size:1.3rem !important; }}
    [data-testid="stMetric"] {{ min-height:90px; padding:.75rem; }}
    button[data-baseweb="tab"] {{ padding:.65rem; }}
}}

/* Reusable product structure, shared across every page. */
.workspace-header {{
    display:flex; align-items:center; justify-content:space-between; gap:1rem;
    padding:.5rem 0 1.25rem; margin-bottom:.65rem; border-bottom:1px solid {BORDER};
}}
.workspace-eyebrow {{ color:{TEXT_DIM}; font-size:.65rem; letter-spacing:.09em; text-transform:uppercase; margin-bottom:.3rem; }}
[data-testid="stMarkdownContainer"] h1.page-title {{ color:{TEXT_HI}; font-size:1.75rem !important; font-weight:600 !important; margin:0 !important; padding:0 !important; line-height:1.35 !important; }}
[data-testid="stMarkdownContainer"] p.page-sub {{ color:{TEXT_DIM}; font-size:.85rem !important; line-height:1.6; margin:.35rem 0 0 !important; max-width:850px; }}
.bt-brand {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.5rem 0 1.25rem; border-bottom:1px solid {BORDER}; }}
.bt-brand-mark {{ width:42px; height:42px; background:{ACCENT}; border-radius:5px; display:flex; align-items:center; justify-content:center; color:white; font-size:.75rem; font-weight:700; }}
.bt-brand-title {{ font-size:1.55rem; color:{TEXT_HI}; font-weight:600; }}
.bt-brand-sub {{ font-size:.71rem; color:{TEXT_DIM}; margin-top:.2rem; }}
.bt-panel, .bt-command {{ background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:8px; padding:.85rem; margin-bottom:.4rem; }}
.bt-section-title {{ color:{TEXT}; font-size:.75rem; font-weight:600; letter-spacing:.035em; margin:0 0 .7rem; }}
.live-bar {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; padding:.5rem .7rem; font-size:.75rem; }}
.live-bar > span:last-child {{ flex-wrap:wrap; }}
.badge, .stage-pill {{ display:inline-flex; align-items:center; padding:3px 7px; border-radius:3px; font-size:.65rem; font-weight:500; white-space:nowrap; }}
.badge-gray, .stage-IDLE, .stage-SLEEP, .stage-STOPPED {{ background:{BG_RAISED}; color:{TEXT_DIM}; }}
.badge-blue, .stage-FETCH, .stage-INDICATORS, .stage-RAG, .stage-AI {{ background:rgba(41,98,255,.12); color:{CYAN}; }}
.badge-green, .stage-DECISION, .stage-EXECUTE {{ background:rgba(38,166,154,.12); color:{C_BUY}; }}
.badge-red, .stage-ERROR {{ background:rgba(239,83,80,.12); color:{C_SELL}; }}
.badge-amber, .stage-RISK {{ background:rgba(247,166,0,.12); color:{AMBER}; }}
.kpi-strip {{ display:flex; flex-wrap:wrap; border:1px solid {BORDER}; background:{BG_DEEP}; border-radius:8px; }}
.kpi-item {{ flex:1; min-width:100px; padding:.95rem 1rem; border-right:1px solid {BORDER}; display:flex; flex-direction:column; gap:.25rem; }}
.kpi-item:last-child {{ border-right:0; }}
.kpi-label {{ font-size:.66rem; color:{TEXT_DIM}; }}
.kpi-value {{ font-size:1.3rem; color:{TEXT}; font-weight:600; font-variant-numeric:tabular-nums; }}
.kpi-sub {{ font-size:.65rem; color:{TEXT_DIM}; }}
.quote-bar {{ display:flex; gap:.65rem; align-items:center; flex-wrap:wrap; padding:.6rem .75rem; border:1px solid {BORDER}; border-bottom:0; background:{BG_PANEL}; }}
.quote-symbol {{ font-size:.9rem; font-weight:600; color:{TEXT_HI}; }}
.quote-detail {{ font-size:.7rem; color:{TEXT_DIM}; }}
.empty-workspace {{ min-height:360px; border:1px solid {BORDER}; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:2rem; background:{BG_DEEP}; }}
.empty-workspace .empty-symbol {{ font-size:2.2rem; color:{BORDER_HI}; margin-bottom:1rem; }}
.empty-workspace h3 {{ font-size:1rem; margin:0 0 .5rem; }}
.empty-workspace p {{ font-size:.8rem; color:{TEXT_DIM}; max-width:380px; line-height:1.6; }}
.watch-row {{ display:grid; grid-template-columns:1fr auto; gap:.3rem; padding:.65rem 0; border-bottom:1px solid {BORDER}; font-size:.78rem; }}
.watch-row small {{ color:{TEXT_DIM}; font-size:.65rem; }}
.watch-price {{ font-variant-numeric:tabular-nums; color:{TEXT}; }}
.desk-facts {{ display:grid; grid-template-columns:1fr auto; gap:.6rem; font-size:.75rem; color:{TEXT_DIM}; }}
.desk-facts b {{ color:{TEXT}; font-weight:500; text-align:right; }}
.sig-buy {{ color:{C_BUY}; }} .sig-sell {{ color:{C_SELL}; }} .sig-hold {{ color:{C_HOLD}; }}
.pulse-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; background:{TEXT_DIM}; flex-shrink:0; }}
.pulse-dot.on {{ background:{C_BUY}; }} .pulse-dot.ai {{ background:{CYAN}; }}
.pulse-dot.exec {{ background:{AMBER}; }} .pulse-dot.err {{ background:{C_SELL}; }}
.log-row {{ font-family:var(--font-mono); font-size:.72rem; padding:6px 8px; border-bottom:1px solid {BORDER}; }}
.log-ts {{ color:{TEXT_DIM}; margin-right:.5rem; }}
.log-lvl {{ display:inline-block; min-width:70px; margin-right:.5rem; }}
.log-msg {{ color:{TEXT}; }}
.log-INFO .log-lvl {{ color:{TEXT_DIM}; }} .log-DECISION .log-lvl {{ color:{CYAN}; }}
.log-TRADE .log-lvl {{ color:{C_BUY}; }} .log-WARN .log-lvl {{ color:{AMBER}; }} .log-ERROR .log-lvl {{ color:{C_SELL}; }}
::-webkit-scrollbar {{ width:7px; height:7px; }}
::-webkit-scrollbar-thumb {{ background:{BORDER_HI}; border-radius:8px; }}
@media (min-width:1100px) {{
    [data-testid="stSidebar"] {{ min-width:236px !important; max-width:236px !important; }}
}}
@media (max-width:760px) {{
    .block-container {{ padding:3rem .7rem 1rem; }}
    .workspace-header, .bt-brand {{ align-items:flex-start; flex-direction:column; gap:.6rem; }}
    .kpi-item {{ flex:1 1 28%; min-width:85px; padding:.6rem; }}
    .kpi-value {{ font-size:.95rem; }}
    .live-bar > span:last-child {{ margin-left:0 !important; gap:.4rem !important; }}
    .empty-workspace {{ min-height:270px; }}
    [data-testid="stMetricValue"] {{ font-size:1.1rem; }}
}}
@media (prefers-reduced-motion:reduce) {{
    *, *::before, *::after {{ animation:none !important; transition:none !important; }}
}}
</style>
"""


def register_chart_theme() -> None:
    """Register once per process; pages select this named template explicitly."""
    import plotly.graph_objects as go
    import plotly.io as pio
    if "bottrade" in pio.templates:
        return
    pio.templates["bottrade"] = go.layout.Template(layout=dict(
        paper_bgcolor=BG_DEEP, plot_bgcolor=BG_DEEP,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=TEXT),
        colorway=[CYAN, C_BUY, AMBER, "#ab47bc", C_SELL],
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER,
                   tickfont=dict(color=TEXT_DIM), showspikes=True,
                   spikecolor=TEXT_DIM, spikethickness=1, spikedash="dot"),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER,
                   tickfont=dict(color=TEXT_DIM), side="right"),
        hoverlabel=dict(bgcolor=BG_RAISED, bordercolor=BORDER_HI, font_color=TEXT),
        modebar=dict(bgcolor=BG_DEEP, color=TEXT_DIM, activecolor=CYAN),
    ))
