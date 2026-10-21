"""
Contenu des onglets "À propos" et "Pipeline de données".

Le Dashboard visualise les données ; cet onglet-ci explique d'où elles viennent.
Les deux sujets restent séparés : aucune des sections du Dashboard ne parle
d'infrastructure, et toute l'explication d'architecture tient ici.

L'architecture d'origine (Snowflake + dbt) a réellement tourné : le premier
dashboard du projet interrogeait directement les tables FINAL du warehouse.
Celui-ci lit les mêmes tables finales, exportées en Parquet sur un VPS.

La circulation de la donnée est rendue en HTML/CSS pur (deux rails, un point
lumineux par connecteur), sans JavaScript ni composant Streamlit custom.

Mise en forme : deux couleurs, et elles portent du sens. Le rail historique est
neutre et en pointillés, le rail actif est à l'accent et en trait plein. Pas
d'icône, pas de dégradé — les étapes sont numérotées.
"""
import re
from urllib.parse import quote

import streamlit as st

import nyc_theme as theme

REPO_URL = "https://github.com/DVDJNBR/NYC_TAXI_PIPELINE"

# Volumétrie — mesurée sur les Parquet, pas arrondie au jugé.
# ROWS_RAW   : SELECT count(*) FROM read_parquet('data/yellow_taxi/*.parquet')
#              (cohérent avec reports/raw_data_quality_report.md)
# ROWS_CLEAN : même compte après les filtres de clean_trips (dont le plafond
#              de 100 mi sur trip_distance) ; c'est le total que le Dashboard
#              affiche.
# Les deux chiffres sont distincts : 77 M est le volume brut ingéré, 74 M le
# volume exploitable. Les confondre fait passer 3 M de lignes écartées pour
# une erreur d'arrondi.
ROWS_RAW   = 76_977_173
ROWS_CLEAN = 73_979_119
PERIOD     = "janvier 2024 – septembre 2025"


def _html(s: str) -> str:
    """Dédente le HTML multi-lignes pour éviter que Streamlit ne le lise comme un bloc de code Markdown."""
    return re.sub(r"^[ \t]+", "", s, flags=re.MULTILINE)


def _n(v: int) -> str:
    """Séparateur de milliers insécable — évite une césure en plein nombre."""
    return f"{v:,}".replace(",", "&nbsp;")


# dbt est le seul outil de la pile absent de simple-icons : les slugs dbt,
# dbtlabs, dbt-labs, getdbt et dbtcloud ne rendent aucun logo, et le dépôt
# simple-icons renvoie 404 sur icons/dbt.svg. shields.io accepte en revanche un
# logo personnalisé en data URI, d'où ce tracé extrait du favicon officiel de
# docs.getdbt.com, recoloré en blanc sur fond transparent.
_DBT_LOGO = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEzLjUgMi41QzEzLjggMi44IDE0IDMuMSAxNCAzLjVDMTQgMy43IDE0IDMuOCAxMy45IDRDMTMuOCA0LjIgMTIuNiA2LjMgMTIuMiA2LjlDMTIgNy4yIDExLjkgNy42IDExLjkgOEMxMS45IDguNCAxMiA4LjggMTIuMiA5LjFDMTIuNiA5LjcgMTMuOCAxMS44IDEzLjkgMTJDMTQgMTIuMiAxNCAxMi4zIDE0IDEyLjVDMTQgMTIuOSAxMy44IDEzLjIgMTMuNSAxMy41QzEzLjIgMTMuNyAxMi45IDEzLjkgMTIuNSAxNEMxMi4zIDE0IDEyLjIgMTMuOSAxMiAxMy44QzExLjggMTMuNyA5LjcgMTIuNiA5LjEgMTIuMkM5LjEgMTIuMiA5IDEyLjEgOSAxMi4xTDYuMSAxMC40QzYuMiAxMSA2LjQgMTEuNSA2LjggMTEuOUM2LjkgMTIgNyAxMiA3LjEgMTIuMUM3IDEyLjEgNi45IDEyLjIgNi45IDEyLjJDNi4zIDEyLjYgNC4yIDEzLjggNCAxMy45QzMuOCAxNCAzLjcgMTQgMy41IDE0QzMuMSAxNCAyLjggMTMuOCAyLjUgMTMuNUMyLjIgMTMuMiAyIDEyLjkgMiAxMi41QzIgMTIuMyAyLjEgMTIuMiAyLjEgMTJDMi4yIDExLjggMy40IDkuNyAzLjggOS4xQzQgOC44IDQuMSA4LjQgNC4xIDhDNC4xIDcuNiA0IDcuMiAzLjggNi45QzMuNCA2LjMgMi4yIDQuMiAyLjEgNEMyLjEgMy44IDIgMy43IDIgMy41QzIgMy4xIDIuMiAyLjggMi41IDIuNUMyLjggMi4yIDMuMSAyIDMuNSAyQzMuNyAyIDMuOCAyLjEgNCAyLjFDNC4yIDIuMiA1LjcgMy4xIDYuNSAzLjZMNi43IDMuN0M2LjggMy43IDYuOCAzLjggNi45IDMuOEw2LjkgMy44TDkuOCA1LjZDOS44IDQuOSA5LjQgNC4zIDguOSAzLjlDOSAzLjkgOS4xIDMuOCA5LjEgMy44QzkuNyAzLjQgMTEuOCAyLjIgMTIgMi4xQzEyLjEgMi4xIDEyLjMgMiAxMi41IDJDMTIuOSAyIDEzLjIgMi4yIDEzLjUgMi41SDEzLjVaTTguMiA4LjhMOC44IDguMkM4LjkgOC4xIDguOSA3LjkgOC44IDcuOEw4LjIgNy4yQzguMSA3LjEgNy45IDcuMSA3LjggNy4yTDcuMiA3LjhDNy4xIDcuOSA3LjEgOC4xIDcuMiA4LjJMNy44IDguOEM3LjkgOC44IDguMSA4LjggOC4yIDguOFpNMTMuNSAyLjVDMTMuOCAyLjggMTQgMy4xIDE0IDMuNUMxNCAzLjcgMTQgMy44IDEzLjkgNEMxMy44IDQuMiAxMi42IDYuMyAxMi4yIDYuOUMxMiA3LjIgMTEuOSA3LjYgMTEuOSA4QzExLjkgOC40IDEyIDguOCAxMi4yIDkuMUMxMi42IDkuNyAxMy44IDExLjggMTMuOSAxMkMxNCAxMi4yIDE0IDEyLjMgMTQgMTIuNUMxNCAxMi45IDEzLjggMTMuMiAxMy41IDEzLjVDMTMuMiAxMy43IDEyLjkgMTMuOSAxMi41IDE0QzEyLjMgMTQgMTIuMiAxMy45IDEyIDEzLjhDMTEuOCAxMy43IDkuNyAxMi42IDkuMSAxMi4yQzkuMSAxMi4yIDkgMTIuMSA5IDEyLjFMNi4xIDEwLjRDNi4yIDExIDYuNCAxMS41IDYuOCAxMS45QzYuOSAxMiA3IDEyIDcuMSAxMi4xQzcgMTIuMSA2LjkgMTIuMiA2LjkgMTIuMkM2LjMgMTIuNiA0LjIgMTMuOCA0IDEzLjlDMy44IDE0IDMuNyAxNCAzLjUgMTRDMy4xIDE0IDIuOCAxMy44IDIuNSAxMy41QzIuMiAxMy4yIDIgMTIuOSAyIDEyLjVDMiAxMi4zIDIuMSAxMi4yIDIuMSAxMkMyLjIgMTEuOCAzLjQgOS43IDMuOCA5LjFDNCA4LjggNC4xIDguNCA0LjEgOEM0LjEgNy42IDQgNy4yIDMuOCA2LjlDMy40IDYuMyAyLjIgNC4yIDIuMSA0QzIuMSAzLjggMiAzLjcgMiAzLjVDMiAzLjEgMi4yIDIuOCAyLjUgMi41QzIuOCAyLjIgMy4xIDIgMy41IDJDMy43IDIgMy44IDIuMSA0IDIuMUM0LjIgMi4yIDUuNyAzLjEgNi41IDMuNkw2LjcgMy43QzYuOCAzLjcgNi44IDMuOCA2LjkgMy44TDYuOSAzLjhMOS44IDUuNkM5LjggNC45IDkuNCA0LjMgOC45IDMuOUM5IDMuOSA5LjEgMy44IDkuMSAzLjhDOS43IDMuNCAxMS44IDIuMiAxMiAyLjFDMTIuMSAyLjEgMTIuMyAyIDEyLjUgMkMxMi45IDIgMTMuMiAyLjIgMTMuNSAyLjVIMTMuNVpNOC4yIDguOEw4LjggOC4yQzguOSA4LjEgOC45IDcuOSA4LjggNy44TDguMiA3LjJDOC4xIDcuMSA3LjkgNy4xIDcuOCA3LjJMNy4yIDcuOEM3LjEgNy45IDcuMSA4LjEgNy4yIDguMkw3LjggOC44QzcuOSA4LjggOC4xIDguOCA4LjIgOC44WiIvPjwvc3ZnPg=="
)

# Badges shields.io, format README GitHub. (libellé, couleur, logo, couleur du logo)
# Le logo est un slug simple-icons, sauf pour dbt où c'est le data URI ci-dessus.
TECH_STACK = [
    ("Python",    "3776AB", "python",        "white"),
    ("Snowflake", "29B5E8", "snowflake",     "white"),
    ("dbt",       "FF694B", _DBT_LOGO,       "white"),
    ("DuckDB",    "FFF000", "duckdb",        "000000"),
    ("pandas",    "150458", "pandas",        "white"),
    ("Parquet",   "50ABF1", "apacheparquet", "000000"),
    ("Streamlit", "FF4B4B", "streamlit",     "white"),
    ("Plotly",    "3F4F75", "plotly",        "white"),
    ("Docker",    "2496ED", "docker",        "white"),
]

FINAL_TABLES = [
    ("daily_summary", "Agrégat",
     "pickup_date · total_trips · total_revenue · avg_distance · avg_fare · avg_tip_pct"),
    ("hourly_patterns", "Agrégat",
     "pickup_hour · total_trips · total_revenue · avg_fare · avg_tip_pct · tranche"),
    ("zone_analysis", "Agrégat",
     "pickup_zone · total_trips · total_revenue · avg_fare · popularity_rank"),
    ("clean_trips", "Staging",
     "trajets filtrés (0 < distance ≤ 100 mi, montants > 0, microsecondes corrigées)"),
]


# ---------------------------------------------------------------------------
# Onglet "À propos"
# ---------------------------------------------------------------------------
PROJECT_FACTS = [
    ("Projet", "Projet étudiant — pipeline de données bout-en-bout"),
    ("Objectif", "Éprouver Snowflake et dbt Core sur un volume réaliste"),
    ("Données", f"{_n(ROWS_RAW)} courses de taxis jaunes new-yorkais<br>"
                f"<span class='fact-sub'>NYC TLC Open Data · {PERIOD}</span>"),
    ("Architecture", "RAW → STAGING → FINAL<br>"
                     "<span class='fact-sub'>modèles dbt versionnés, testés à chaque build</span>"),
    ("Restitution", "Streamlit + Plotly"),
]

NEXT_STEPS = [
    ("Pipeline de données", "le détail de la chaîne, étape par étape"),
    ("Dashboard", "l'analyse menée sur les tables finales"),
]


def render_about_tab():
    theme_p = theme.palette()
    st.markdown(_html(f"""
    <style>
      .fact-row {{ display:flex; gap:20px; padding:11px 0;
                   border-bottom:1px solid {theme_p['RULE']}; }}
      .fact-key {{ flex:0 0 118px; font-size:0.66rem; font-weight:600;
                   letter-spacing:0.09em; text-transform:uppercase;
                   color:{theme_p['TEXT_DIM']}; padding-top:3px; }}
      .fact-val {{ flex:1; font-size:0.92rem; line-height:1.5;
                   color:{theme_p['TEXT_LIGHT']}; }}
      .fact-sub {{ font-size:0.78rem; color:{theme_p['TEXT_DIM']}; }}
      .next-row {{ display:flex; gap:14px; padding:7px 0; font-size:0.84rem; }}
      .next-key {{ flex:0 0 170px; color:{theme_p['ACCENT']}; font-weight:600; }}
      .next-val {{ color:{theme_p['TEXT_DIM']}; }}
    </style>
    """), unsafe_allow_html=True)

    rows = "".join(
        f'<div class="fact-row"><div class="fact-key">{key}</div>'
        f'<div class="fact-val">{value}</div></div>'
        for key, value in PROJECT_FACTS
    )
    st.markdown(_html(f'<div style="max-width:68ch">{rows}</div>'), unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    badges = "".join(
        f'<img src="https://img.shields.io/badge/{label}-{color}'
        f'?style=flat-square&logo={quote(logo, safe="")}&logoColor={logo_text}" '
        f'alt="{label}" style="margin:0 5px 6px 0; height:22px;">'
        for label, color, logo, logo_text in TECH_STACK
    )
    st.markdown(_html(f"<div>{badges}</div>"), unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    steps = "".join(
        f'<div class="next-row"><span class="next-key">→ {key}</span>'
        f'<span class="next-val">{value}</span></div>'
        for key, value in NEXT_STEPS
    )
    st.markdown(_html(f'<div style="max-width:68ch">{steps}</div>'), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(f"[Code source sur GitHub]({REPO_URL})")


# ---------------------------------------------------------------------------
# Onglet "Pipeline de données" — circulation animée
# ---------------------------------------------------------------------------
# Chaque étage porte une vignette montrant un extrait réel de ce qu'il contient,
# pour que le schéma ne soit pas qu'un enchaînement de boîtes.

_P_SOURCE = """<div class="flow-path">yellow_tripdata_2024-06.parquet</div>
<pre class="flow-json">{
  "tpep_pickup_datetime": "2024-06-12 18:04:21",
  "trip_distance": 3.80,
  "PULocationID": 161,
  "payment_type": 1,
  "total_amount": 27.95
}</pre>"""

_P_RAW = f"""<div class="flow-path">NYC_TAXI_DB.RAW.yellow_tripdata</div>
<div class="flow-kv"><span>lignes</span><b>{_n(ROWS_RAW)}</b></div>
<div class="flow-kv"><span>charge</span><b>COPY INTO</b></div>
<div class="flow-note">Copié tel quel, horodatages aberrants compris
(2002 → 2026 dans la source TLC).</div>"""

_P_STAGING = f"""<div class="flow-path">STAGING.clean_trips &middot; dbt Core</div>
<pre class="flow-json">WHERE trip_distance &gt; 0
  AND trip_distance &lt;= 100
  AND total_amount  &gt; 0
  AND pickup &lt; dropoff</pre>
<div class="flow-kv"><span>retenues</span><b>{_n(ROWS_CLEAN)}</b></div>
<div class="flow-note">&minus;{_n(ROWS_RAW - ROWS_CLEAN)} lignes écartées, dont
3&nbsp;700 courses de plus de 100 mi (jusqu'à 398&nbsp;000 mi : erreurs GPS qui
doublaient la distance moyenne). + tests dbt_utils.</div>"""

_P_FINAL = """<div class="flow-path">FINAL.daily_summary</div>
<div class="flow-kv"><span>pickup_date</span><b>2024-06-12</b></div>
<div class="flow-kv"><span>total_trips</span><b>118&nbsp;402</b></div>
<div class="flow-kv"><span>avg_fare</span><b>$28.14</b></div>
<div class="flow-kv"><span>avg_tip_pct</span><b>18.6%</b></div>"""

_P_COPY = """<div class="flow-path">./data/yellow_taxi/*.parquet</div>
<div class="flow-kv"><span>fichiers</span><b>21</b></div>
<div class="flow-kv"><span>montage</span><b>bind-mount</b></div>
<div class="flow-note">Téléchargés une fois par entrypoint.sh.</div>"""

_P_DUCKDB = """<div class="flow-path">Scan en mémoire</div>
<pre class="flow-json">read_parquet(
  'data/yellow_taxi/*.parquet'
)</pre>
<div class="flow-note">Mêmes requêtes, pas de warehouse.</div>"""

_P_CACHE = """<div class="flow-path">./data/aggregates/*.parquet</div>
<div class="flow-kv"><span>daily</span><b>37 Ko</b></div>
<div class="flow-kv"><span>hourly</span><b>6 Ko</b></div>
<div class="flow-kv"><span>zones</span><b>9 Ko</b></div>
<div class="flow-note">Persisté : un redémarrage ne rescanne pas.</div>"""

_P_HERE = """<div class="flow-here">Vous êtes ici</div>
<div class="flow-path">streamlit_dashboard_local.py</div>
<div class="flow-note">Streamlit + Plotly, requêtes mises en cache.</div>"""

_P_OFFLINE = """<div class="flow-path">Dashboard sur Snowflake</div>
<pre class="flow-json">SELECT pickup_date, total_trips,
       total_revenue, avg_distance
FROM daily_summary</pre>
<div class="flow-note">Le dashboard lisait les tables FINAL
directement dans le warehouse.</div>"""

# (titre, sous-titre, vignette)
RAIL_SNOWFLAKE = [
    ("NYC TLC",   "Open Data",     _P_SOURCE),
    ("RAW",       "Snowflake",     _P_RAW),
    ("STAGING",   "dbt Core",      _P_STAGING),
    ("FINAL",     "Marts agrégés", _P_FINAL),
    ("Dashboard", "Streamlit",     _P_OFFLINE),
]

RAIL_DUCKDB = [
    ("NYC TLC",   "Mêmes Parquet", _P_SOURCE),
    ("Copie VPS", "Volume Docker", _P_COPY),
    ("DuckDB",    "read_parquet()", _P_DUCKDB),
    ("Agrégats",  "Cache disque",  _P_CACHE),
    ("Dashboard", "Streamlit",     _P_HERE),
]

# Durée d'un cycle complet : ~1.1 s par étage franchi.
_CYCLE = round((len(RAIL_SNOWFLAKE) - 1) * 1.1, 2)

_FLOW_CSS = """
<style>
  .flow-wrap  { overflow-x:auto; padding:4px 2px 8px; }
  .flow-row   { display:flex; align-items:flex-start; min-width:max-content; }
  .flow-col   { display:flex; flex-direction:column; align-items:stretch; gap:6px;
                width:146px; flex:0 0 auto; }
  .flow-node  { box-sizing:border-box; padding:10px 12px; border-radius:2px;
                border:1px solid var(--flow-color);
                border-left:2px solid var(--flow-color);
                animation:flowPulse __T__s ease-in-out infinite; }
  .flow-step  { font-size:0.62rem; font-weight:600; letter-spacing:0.08em;
                color:var(--flow-color); }
  .flow-node-title { font-weight:600; font-size:0.84rem; color:TEXTLIGHT; margin-top:2px; }
  .flow-node-sub   { font-size:0.68rem; color:TEXTDIM; margin-top:1px; }
  .flow-preview { box-sizing:border-box; min-height:108px; padding:8px 10px;
                  border-left:1px solid CARDBORDER;
                  font-size:0.62rem; color:TEXTDIM; overflow:hidden; }
  .flow-path  { font-size:0.6rem; color:var(--flow-color); margin-bottom:4px; word-break:break-all; }
  .flow-json  { margin:0; font-size:0.6rem; line-height:1.45; white-space:pre-wrap;
                word-break:break-word; color:TEXTDIM; }
  .flow-note  { margin-top:5px; opacity:0.75; }
  .flow-kv    { display:flex; justify-content:space-between; gap:6px; padding:2px 0;
                border-bottom:1px solid CARDBORDER; }
  .flow-kv:last-child { border-bottom:none; }
  .flow-kv span { color:TEXTDIM; min-width:0; overflow-wrap:break-word; }
  .flow-kv b    { color:TEXTLIGHT; font-weight:600; white-space:nowrap; flex:0 0 auto;
                  font-variant-numeric:tabular-nums; }
  .flow-here  { color:ACCENTCOL; font-weight:600; font-size:0.68rem;
                letter-spacing:0.06em; text-transform:uppercase; margin-bottom:3px; }

  .flow-connector      { position:relative; width:30px; height:18px; margin-top:20px; flex:0 0 auto; }
  .flow-connector-line { position:absolute; top:50%; left:0; right:0; height:1px;
                         background:CARDBORDER; transform:translateY(-50%); }
  .flow-connector-spark{ position:absolute; top:50%; left:0; width:5px; height:5px;
                         background:var(--flow-color);
                         transform:translate(-50%,-50%); opacity:0;
                         animation:flowTravel __T__s linear infinite; }

  /* Rail historique : pointillés, pulsation deux fois plus lente, moins présent.
     Il se lit comme "monté, puis éteint" sans avoir besoin d'une légende. */
  .rail-past .flow-node           { border-style:dashed; opacity:0.72; animation-duration:__T2__s; }
  .rail-past .flow-connector-line { background:repeating-linear-gradient(90deg,
                                      CARDBORDER 0 4px, transparent 4px 8px); }
  .rail-past .flow-connector-spark{ animation-duration:__T2__s; }

  .rail-box    { border-top:1px solid CARDBORDER; padding:18px 0 4px;
                 position:relative; margin:0 0 26px; }
  .rail-head   { display:flex; justify-content:space-between; align-items:baseline;
                 gap:16px; margin-bottom:12px; flex-wrap:wrap; }
  .rail-label  { font-size:0.72rem; font-weight:600; letter-spacing:0.06em;
                 text-transform:uppercase; }
  .rail-badge  { font-size:0.66rem; }

  @keyframes flowPulse {
      0%        { background:transparent; }
      2%        { background:color-mix(in srgb, var(--flow-color) 12%, transparent); }
      SLICEm2   { background:color-mix(in srgb, var(--flow-color) 12%, transparent); }
      SLICEpct  { background:transparent; }
      100%      { background:transparent; }
  }
  @keyframes flowTravel {
      0%        { left:0%;   opacity:0; }
      6%        { opacity:1; }
      SLICEm3   { left:100%; opacity:1; }
      SLICEpct, 100% { left:100%; opacity:0; }
  }
  @media (prefers-reduced-motion: reduce) {
      .flow-node, .flow-connector-spark { animation:none !important; }
  }
</style>
"""


def _flow_rail(stages, color, rail_class=""):
    """Rend une file d'étages reliés par des connecteurs animés.

    Le délai négatif décale chaque étage d'un cran dans le cycle : la pulsation
    et le point lumineux se propagent donc de gauche à droite, en continu.
    """
    n = len(stages)
    step = _CYCLE / (n - 1)

    cols = []
    for i, (title, sub, preview) in enumerate(stages):
        cols.append(f"""
            <div class="flow-col" style="--flow-color:{color};">
              <div class="flow-node" style="animation-delay:{-round(i * step, 2)}s;">
                <div class="flow-step">{i + 1:02d}</div>
                <div class="flow-node-title">{title}</div>
                <div class="flow-node-sub">{sub}</div>
              </div>
              <div class="flow-preview">{preview}</div>
            </div>
        """)

    conns = []
    for i in range(n - 1):
        conns.append(f"""
            <div class="flow-connector" style="--flow-color:{color};">
              <span class="flow-connector-line"></span>
              <span class="flow-connector-spark"
                    style="animation-delay:{-round(i * step, 2)}s;"></span>
            </div>
        """)

    row = cols[0]
    for i in range(1, n):
        row += conns[i - 1] + cols[i]
    return f'<div class="flow-row {rail_class}">{row}</div>'


def _flow_css():
    slice_pct = 100 / (len(RAIL_SNOWFLAKE) - 1)
    return (
        _FLOW_CSS
        .replace("__T2__", str(round(_CYCLE * 2, 2)))
        .replace("__T__", str(_CYCLE))
        .replace("SLICEm2", f"{max(0.1, slice_pct - 2):.2f}%")
        .replace("SLICEm3", f"{max(0.1, slice_pct - 3):.2f}%")
        .replace("SLICEpct", f"{slice_pct:.2f}%")
        .replace("TEXTLIGHT", theme.TEXT_LIGHT)
        .replace("TEXTDIM", theme.TEXT_DIM)
        .replace("CARDBORDER", theme.RULE)
        .replace("ACCENTCOL", theme.ACCENT)
    )


def render_pipeline_tab():
    st.markdown(_flow_css(), unsafe_allow_html=True)

    st.header("Circulation de la donnée")
    st.markdown(_html(f"""
    <p style="color:{theme.TEXT_DIM}; font-size:0.85rem; line-height:1.65;
              max-width:88ch; margin:0 0 22px;">
      Deux chemins partent de la même source. Le premier est l'architecture conçue pour le
      projet, aujourd'hui éteinte ; le second alimente réellement les graphiques de l'onglet
      Dashboard. Les étages logiques sont les mêmes — seul le moteur de calcul change.
      Le point lumineux indique le sens de circulation ; chaque vignette montre un extrait
      réel de ce que contient l'étage.
    </p>
    """), unsafe_allow_html=True)

    st.markdown(_html(f"""
    <div class="rail-box">
      <div class="rail-head">
        <span class="rail-label" style="color:{theme.NEUTRAL};">Architecture d'origine — Snowflake + dbt Core</span>
        <span class="rail-badge" style="color:{theme.WORSE};">Warehouse arrêté</span>
      </div>
      <div class="flow-wrap">{_flow_rail(RAIL_SNOWFLAKE, theme.NEUTRAL, "rail-past")}</div>
    </div>

    <div class="rail-box">
      <div class="rail-head">
        <span class="rail-label" style="color:{theme.ACCENT};">Chemin actif — copie Parquet sur VPS + DuckDB</span>
        <span class="rail-badge" style="color:{theme.BETTER};">En service</span>
      </div>
      <div class="flow-wrap">{_flow_rail(RAIL_DUCKDB, theme.ACCENT)}</div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(_html(f"""
    <p style="color:{theme.TEXT_DIM}; font-size:0.82rem; line-height:1.65;
              max-width:88ch; margin:0 0 30px;">
      <b style="color:{theme.TEXT_LIGHT};">Pourquoi deux rails ?</b> Le warehouse Snowflake
      était facturé à l'usage et n'est plus financé depuis la fin de la formation. Les tables
      <code>FINAL</code> ont donc été exportées en Parquet et déposées sur un VPS loué.
      Les étages logiques n'ont pas bougé — ingestion, nettoyage, agrégation, restitution —
      seul le moteur qui les exécute a changé. Les chiffres du Dashboard sont les mêmes
      qu'en sortie de warehouse.
    </p>
    """), unsafe_allow_html=True)

    st.header("Schéma FINAL — tables analytiques")
    rows = "".join(
        _html(f"""
        <div style="display:flex; flex-wrap:wrap; align-items:baseline; gap:14px;
                    padding:9px 0; border-bottom:1px solid {theme.RULE};">
          <span style="flex:0 0 130px; font-weight:600; color:{theme.TEXT_LIGHT};
                       font-family:monospace; font-size:0.82rem;">{name}</span>
          <span style="flex:0 0 58px; font-size:0.68rem; letter-spacing:0.06em;
                       text-transform:uppercase; color:{theme.TEXT_DIM};">{kind}</span>
          <span style="color:{theme.TEXT_DIM}; font-size:0.74rem; font-family:monospace;">{cols}</span>
        </div>
        """)
        for name, kind, cols in FINAL_TABLES
    )
    st.markdown(_html(f"<div>{rows}</div>"), unsafe_allow_html=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    st.header("Les deux infrastructures")
    c1, c2 = st.columns(2)
    for col, (title, state, state_color, items) in zip(
        (c1, c2),
        (
            ("Snowflake + dbt", "Architecture d'origine", theme.NEUTRAL, [
                "Warehouse <code>NYC_TAXI_WH</code>, DB <code>NYC_TAXI_DB</code>",
                "Schémas <code>RAW</code> / <code>STAGING</code> / <code>FINAL</code>",
                "Chargement par <code>COPY INTO</code> depuis les Parquet TLC",
                "Modèles dbt + tests automatiques (dbt_utils)",
            ]),
            ("DuckDB + Docker (VPS)", "En service", theme.BETTER, [
                "Tables <code>FINAL</code> exportées en Parquet",
                "Scan direct des fichiers, sans base distante",
                "Conteneur Docker derrière Caddy (reverse proxy)",
                "Volume bind-mount, téléchargement au 1ᵉʳ démarrage",
            ]),
        ),
    ):
        with col:
            lis = "".join(f"<li>{i}</li>" for i in items)
            st.markdown(_html(f"""
            <div style="border-top:1px solid {theme.RULE}; padding:14px 0 0;">
              <div style="display:flex; justify-content:space-between; align-items:baseline; gap:10px;">
                <span style="font-weight:600; color:{theme.TEXT_LIGHT}; font-size:0.9rem;">{title}</span>
                <span style="font-size:0.66rem; color:{state_color};">{state}</span>
              </div>
              <ul style="color:{theme.TEXT_DIM}; font-size:0.78rem; margin:10px 0 0;
                         padding-left:16px; line-height:1.8;">{lis}</ul>
            </div>
            """), unsafe_allow_html=True)
