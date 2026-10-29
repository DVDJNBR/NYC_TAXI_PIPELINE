"""
Contenu des onglets "À propos" et "Pipeline de données".

Le Dashboard visualise les données ; cet onglet-ci explique d'où elles viennent.
Les deux sujets restent séparés : aucune des sections du Dashboard ne parle
d'infrastructure, et toute l'explication d'architecture tient ici.

La page ne montre qu'une chose : le chemin que prend la donnée, de la source
publique jusqu'à l'écran. Six étages, reliés par des connecteurs où court un
point lumineux, en HTML/CSS pur — ni JavaScript, ni composant Streamlit
custom.

Chaque étage porte un extrait réel de ce qu'il contient : un chemin de
fichier, des colonnes, un filtre. Pas de commentaire sur l'architecture, pas
de comparaison, pas de verdict : le schéma nomme ce qui existe dans le dépôt
et s'arrête là. Pas d'icône, pas de dégradé — les étages sont numérotés.

Aucun texte affiché n'est écrit ici : tout vient de `nyc_lang`, par clé. Ce qui
reste en dur, ce sont les noms de tables, de colonnes, de fichiers et de
produits — les traduire ferait mentir le schéma, qui doit nommer ce qui existe
réellement dans le dépôt. Conséquence : les blocs de contenu sont des fonctions
et non plus des constantes de module, puisqu'ils dépendent de la langue du
visiteur et non de l'état du serveur au démarrage.
"""
import json
import pathlib
import re
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

import nyc_lang
import nyc_theme as theme

from nyc_lang import t

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
# Les deux bornes citées dans la vignette STAGING : le nombre de courses
# au-delà du plafond de 100 mi, et la pire d'entre elles.
OUTLIERS   = 3_700
MAX_MILES  = 398_000


def _html(s: str) -> str:
    """Dédente le HTML multi-lignes pour éviter que Streamlit ne le lise comme un bloc de code Markdown."""
    return re.sub(r"^[ \t]+", "", s, flags=re.MULTILINE)


def _n(v: int) -> str:
    """Entier groupé selon la langue — virgule en anglais, insécable sinon."""
    return nyc_lang.n(v)


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


# ---------------------------------------------------------------------------
# Onglet "À propos"
# ---------------------------------------------------------------------------
def _project_facts():
    return [
        (t("about.k.project"), t("about.v.project")),
        (t("about.k.goal"), t("about.v.goal")),
        (t("about.k.data"),
         t("about.v.data", rows=_n(ROWS_RAW)) + "<br>"
         f"<span class='fact-sub'>"
         + t("about.v.data_sub", period=t("about.period")) + "</span>"),
        (t("about.k.arch"),
         "RAW → STAGING → FINAL<br>"
         f"<span class='fact-sub'>{t('about.v.arch_sub')}</span>"),
        (t("about.k.output"), "Streamlit + Plotly"),
    ]


def _next_steps():
    return [
        (t("nav.pipeline"), t("about.next.pipeline")),
        ("Dashboard", t("about.next.dashboard")),
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
        for key, value in _project_facts()
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
        for key, value in _next_steps()
    )
    st.markdown(_html(f'<div style="max-width:68ch">{steps}</div>'), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(f"[{t('about.repo')}]({REPO_URL})")


# ---------------------------------------------------------------------------
# Onglet "Pipeline de données" — le chemin de la donnée, animé
# ---------------------------------------------------------------------------
# Porté du même bloc dans WATT_WATCHER (app_v2/.../PipelineDiagram.jsx et les
# règles .pipeline-* de son index.css), structure et minutage compris.
#
# La forme fait le propos : une colonne de sources à gauche, puis une carte
# par produit, et dans chaque carte les couches qu'il héberge. On lit une
# architecture — qui tient quoi — là où cinq panneaux identiques alignés ne
# donnaient qu'une suite de boîtes. RAW, STAGING et FINAL sont trois schémas
# d'un même entrepôt : ils vivent donc dans la même carte, et leur filet de
# gauche prend les couleurs de médaille, qui sont la convention pour ces
# trois couches.
#
# Une bille parcourt la chaîne segment par segment, allume chaque couche et
# la laisse allumée ; l'aperçu de la couche flashe au passage puis retombe.
# Une couche allumée est une trace du passage ; un aperçu montre ce que la
# donnée vient de devenir, donc il ne reste pas allumé.
#
# Les connecteurs et la position des aperçus sont mesurés, pas écrits en
# dur : une seule couche SVG dont les `d` sont calculés depuis la position
# réelle de chaque bloc. WATT_WATCHER y mettait des coordonnées à la main,
# ce qui tient tant qu'aucun libellé ne change ; ici tout est relevé au
# rendu, donc une police ou une traduction ne décale rien.
#
# La durée d'un segment vient de sa longueur à l'écran divisée par une
# vitesse en pixels par seconde : une longue diagonale met visiblement plus
# de temps qu'un saut court, ce qu'une durée fixe ne rend pas.
#
# La géométrie est en pixels fixes, et la scène défile horizontalement quand
# la fenêtre est trop étroite. C'est le choix du bloc d'origine : un
# diagramme qui se replie par blocs est un autre composant, et replié il ne
# montre plus de chaîne.
_CDN = "https://d37ci6vzurychx.cloudfront.net"
_LOGOS = pathlib.Path(__file__).resolve().parent / "assets" / "logos"


def _mark(slug: str, fallback: str = "") -> str:
    """Le logo d'un produit, inséré en SVG plutôt que servi en <img>.

    Les fichiers vivent dans assets/, qui n'est pas exposé en HTTP — seul
    static/ l'est, et il est hors du dépôt. Inséré dans le balisage, le logo
    n'a besoin d'aucune URL et d'aucun aller-retour réseau au rendu.

    Le fill de marque est remplacé par `currentColor` : un seul tracé sert
    alors les deux thèmes, au lieu d'un couple clair/sombre comme dans le
    bloc d'origine.
    """
    f = _LOGOS / f"{slug}.svg"
    if not f.exists():
        return fallback
    svg = re.sub(r"<title>.*?</title>", "", f.read_text(), flags=re.S)
    return re.sub(r'fill="#[0-9A-Fa-f]{3,8}"', 'fill="currentColor"', svg)


# Tracés maison pour ce que simple-icons ne couvre pas. dbt en est absent
# (les slugs dbt, dbtlabs, getdbt, dbt-labs et dbtcore renvoient tous 404) :
# son tracé vient du favicon officiel de docs.getdbt.com, déjà embarqué plus
# haut en data URI pour les badges du README. Les trois autres sont des
# pictogrammes — un tableau, un polygone, un histogramme — là où un logo
# n'existe pas : les sources sont trois chemins HTTP chez le même émetteur,
# donc c'est le format du fichier qui les distingue, pas la marque.
_G_DBT = (
    '<svg viewBox="0 0 16 16"><path fill="currentColor" d="M13.5 2.5C13.8 2.8'
    " 14 3.1 14 3.5C14 3.7 14 3.8 13.9 4C13.8 4.2 12.6 6.3 12.2 6.9C12 7.2"
    " 11.9 7.6 11.9 8C11.9 8.4 12 8.8 12.2 9.1C12.6 9.7 13.8 11.8 13.9 12C14"
    " 12.2 14 12.3 14 12.5C14 12.9 13.8 13.2 13.5 13.5C13.2 13.7 12.9 13.9"
    " 12.5 14C12.3 14 12.2 13.9 12 13.8C11.8 13.7 9.7 12.6 9.1 12.2C9.1 12.2"
    " 9 12.1 9 12.1L6.1 10.4C6.2 11 6.4 11.5 6.8 11.9C6.9 12 7 12 7.1"
    " 12.1C7 12.1 6.9 12.2 6.9 12.2C6.3 12.6 4.2 13.8 4 13.9C3.8 14 3.7 14"
    " 3.5 14C3.1 14 2.8 13.8 2.5 13.5C2.2 13.2 2 12.9 2 12.5C2 12.3 2.1 12.2"
    " 2.1 12C2.2 11.8 3.4 9.7 3.8 9.1C4 8.8 4.1 8.4 4.1 8C4.1 7.6 4 7.2 3.8"
    " 6.9C3.4 6.3 2.2 4.2 2.1 4C2.1 3.8 2 3.7 2 3.5C2 3.1 2.2 2.8 2.5"
    " 2.5C2.8 2.2 3.1 2 3.5 2C3.7 2 3.8 2.1 4 2.1C4.2 2.2 5.7 3.1 6.5"
    " 3.6L6.7 3.7C6.8 3.7 6.8 3.8 6.9 3.8L6.9 3.8L9.8 5.6C9.8 4.9 9.4 4.3"
    " 8.9 3.9C9 3.9 9.1 3.8 9.1 3.8C9.7 3.4 11.8 2.2 12 2.1C12.1 2.1 12.3 2"
    " 12.5 2C12.9 2 13.2 2.2 13.5 2.5H13.5ZM8.2 8.8L8.8 8.2C8.9 8.1 8.9 7.9"
    ' 8.8 7.8L8.2 7.2C8.1 7.1 7.9 7.1 7.8 7.2L7.2 7.8C7.1 7.9 7.1 8.1 7.2'
    ' 8.2L7.8 8.8C7.9 8.8 8.1 8.8 8.2 8.8Z"/></svg>'
)
_G_GRID = (
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
    ' stroke-width="1.3"><rect x="2" y="2.5" width="12" height="11" rx="1"/>'
    '<path d="M2 6h12M2 10h12M6.5 6v7.5"/></svg>'
)
_G_BARS = (
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
    ' stroke-width="1.5" stroke-linecap="round">'
    '<path d="M3 13V9M6.3 13V4M9.7 13V7M13 13V5.5"/></svg>'
)
_G_STACK = (
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
    ' stroke-width="1.3"><ellipse cx="8" cy="4" rx="5.5" ry="2.2"/>'
    '<path d="M2.5 4v8c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2V4"/>'
    '<path d="M2.5 8.2c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2"/></svg>'
)


# Les trois appels, tels que les scripts les font vraiment : trois chemins
# HTTP sur le CDN de la TLC, un par scripts/B_load_data.py,
# scripts/G_build_aggregates.py et scripts/H_build_geo.py.
#
# `enter` est la couche où la source entre dans la chaîne. Les géométries ne
# passent pas par l'entrepôt : elles sont lues une fois pour construire la
# carte, donc leur connecteur vise les agrégats et non RAW. Une source qui
# partirait de la première couche alors qu'elle n'y passe pas raconterait un
# chemin faux.
def _sources():
    p = theme.palette()
    return [
        {"id": "trips", "label": "Trip Records", "sub": "21 Parquet mensuels",
         "glyph": _mark("apacheparquet", _G_GRID),
         "path": "/trip-data/yellow_tripdata_{YYYY-MM}.parquet",
         "color": p["ACCENT"], "enter": 0},
        {"id": "zones", "label": "Taxi Zones", "sub": "1 CSV · 265 zones",
         "glyph": _G_GRID,
         "path": "/misc/taxi_zone_lookup.csv",
         "color": p["BRICK"], "enter": 0},
    ]


def _table(rows):
    """Un aperçu en mini-table : une colonne par ligne, sa valeur en face.

    Sans bandeau de légende au-dessus. La pastille juste au-dessus nomme
    déjà la couche, et la tige relie les deux : répéter « 77 M lignes » en
    capitales ajoutait un titre à quelque chose qui en avait un.
    """
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="pk-table"><tbody>{body}</tbody></table>'


def _kpi(value, sub):
    """Le dernier aperçu n'est pas un fichier : c'est ce que la chaîne sert.

    Lister les binaires servis au navigateur décrit la plomberie une
    deuxième fois. Un chiffre lu dans le dashboard dit à quoi tout ce
    chemin aboutit. Et c'est le chiffre le plus évident qui convient ici :
    le bloc raconte un transport de données, pas une analyse — un KPI
    malin détournerait le regard du chemin, qui est le sujet.
    """
    return f'<div class="pk-kpi"><b>{value}</b><em>{sub}</em></div>'


def _code(text):
    """Un aperçu en bloc de code, quand la matière de la couche est du SQL."""
    return f'<div class="pk-code"><code>{text}</code></div>'


# Les cinq couches, carte par carte : (produit, logo, [couche...]).
# Une couche est (clé, nom, filet de gauche, icône, aperçu).
#
# L'aperçu est la vraie matière de la couche — une ligne de table, une
# clause SQL, un listing de fichiers. C'est ce que montrait le bloc
# d'origine, et c'est ce qui dit le dégraissage sans qu'on lise une phrase :
# 77 millions de lignes brutes, 3 millions écartées, 639 journées.
def _cards():
    return [
        ("Snowflake", _mark("snowflake", _G_STACK), [
            ("raw", "RAW", "#a9763f", _G_STACK,
             _table([
                 ("tpep_pickup", "2024-06-12 18:04"),
                 ("PULocationID", "161"),
                 ("DOLocationID", "236"),
                 ("trip_distance", "3.80"),
                 ("total_amount", "27.95"),
             ])),
            ("stg", "STAGING · dbt", "#9ea4ad", _G_DBT,
             _code("WHERE trip_distance &gt; 0\n"
                   "  AND trip_distance &lt;= 100\n"
                   "  AND total_amount  &gt; 0\n"
                   "  AND pickup &lt; dropoff")),
            ("final", "FINAL · dbt", "#c9a227", _G_DBT,
             _table([
                 ("pickup_date", "2024-06-12"),
                 ("total_trips", "118 402"),
                 ("total_revenue", "$3 331 830"),
                 ("avg_fare", "$28.14"),
                 ("avg_tip_pct", "18,6 %"),
             ])),
        ]),
        ("Streamlit", _mark("streamlit", _G_BARS), [
            ("app", "Dashboard", "", _G_BARS,
             _kpi("115 773",
                  "courses par jour en moyenne, "
                  "sur les 639 journées servies")),
        ]),
    ]

# Géométrie, en pixels, et elle doit s'additionner : une tuile de source fait
# 132, une carte produit 18 + 134 par couche + 24 de gouttière + 11, la
# gouttière entre cartes 10 et le bus des sources 30. Trois couches chez
# Snowflake, une chez Streamlit :
#
#     132 + 30 + 479 + 10 + 163 = 814
#
# Rien ici ne dépend de la longueur d'un libellé — c'est ce qui permet de
# mesurer les connecteurs au rendu sans qu'ils sautent d'une langue à
# l'autre.
W_SCENE = 814

_PIPE_HTML = r"""
<!doctype html>
<meta charset="utf-8">
<style>
  :root {
      --ink: __INK__; --dim: __DIM__; --rule: __RULE__;
      --bg: __BG__; --panel: __PANEL__;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; }
  body {
      color: var(--ink);
      font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
      -webkit-font-smoothing: antialiased; }

  /* La scène défile en bloc quand la fenêtre est trop étroite. Repliée par
     blocs elle ne montrerait plus de chaîne, et c'est la chaîne qui est le
     sujet. */
  #scroll { width: 100%; overflow-x: auto; padding-bottom: 6px; }
  #scene { position: relative; width: __W__px; margin: 0 auto; }

  #flow { position: absolute; left: 0; top: 0;
          pointer-events: none; overflow: visible; z-index: 1; }
  .chain { stroke: var(--rule); stroke-width: 1; fill: none; }
  .stem  { stroke: var(--rule); stroke-width: 1; fill: none; }
  .trail { stroke-width: 1.4; stroke-linecap: round; fill: none; }
  #bead { position: absolute; left: -5px; top: -5px;
          width: 10px; height: 10px; border-radius: 50%;
          background: var(--flow, var(--ink));
          border: 1px solid var(--bg); opacity: 0; z-index: 1; }

  /* Volontairement non positionnée : un élément statique peint sous tout
     élément positionné, quel que soit le signe du z-index. C'est ce qui
     laisse les connecteurs passer AU-DESSUS du fond d'une carte produit sur
     la portion qui la traverse, et disparaître seulement sous la pastille
     de couche. Positionner la rangée masquerait toute la traversée. */
  #row { display: flex; align-items: center; min-height: 196px; }
  .gap { flex: 0 0 auto; width: 10px; }
  .gap.bus { width: 30px; }

  /* Centrée, pas alignée en haut : les deux tuiles se répartissent de part
     et d'autre de la rangée, donc leurs deux connecteurs convergent sur le
     bord gauche de RAW et dessinent un chevron. Alignées en haut, elles
     envoyaient deux obliques parallèles vers le bas, ce qui ne disait pas
     que les deux sources se rejoignent au même endroit. */
  #srcs { align-self: center; display: flex;
          flex-direction: column; gap: 10px; }
  .p-node {
      display: block; width: 132px; flex: 0 0 auto;
      position: relative; z-index: 2;
      background: var(--panel); border: 1px solid var(--rule);
      border-radius: 4px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
      color: inherit; font: inherit; appearance: none; margin: 0; padding: 0;
      cursor: pointer; text-align: left;
      transition: border-color 0.18s ease, box-shadow 0.18s ease,
                  transform 0.18s ease; }
  .p-node:hover { border-color: var(--src);
                  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35);
                  transform: translateY(-1px); }
  .p-node.on { border-color: var(--src);
               box-shadow: inset 0 0 0 1px var(--src); }
  .p-body { display: flex; align-items: center; gap: 8px;
            padding: 9px 10px; }
  .p-dot { width: 8px; height: 8px; flex: 0 0 auto; border-radius: 50%;
           border: 1.5px solid var(--src); background: transparent; }
  .p-node.on .p-dot { background: var(--src); }
  .p-ico { width: 22px; height: 22px; flex: 0 0 auto;
           display: grid; place-items: center; color: var(--src); }
  .p-ico svg { width: 18px; height: 18px; display: block; }
  .p-name { font-size: 0.74rem; font-weight: 600; line-height: 1.2;
            letter-spacing: -0.005em; }
  .p-sub { display: block; margin-top: 1px; font-size: 0.58rem;
           font-weight: 400; color: var(--dim); letter-spacing: 0; }

  .store { position: relative; flex: 0 0 auto;
           background: var(--panel); border: 1px solid var(--rule);
           border-radius: 4px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3); }
  .store-head { margin: 0; display: flex; align-items: center; gap: 8px;
                padding: 9px 11px; border-bottom: 1px solid var(--rule); }
  .store-head .ico { width: 22px; height: 22px; flex: 0 0 auto;
                     display: grid; place-items: center; color: var(--ink); }
  .store-head .ico svg { width: 18px; height: 18px; display: block; }
  .store-head b { font-size: 0.74rem; font-weight: 600; line-height: 1.2;
                  letter-spacing: -0.005em; }
  .store-list { margin: 0; padding: 9px 11px 11px 18px; list-style: none;
                display: flex; gap: 24px; }

  .p-chip {
      position: relative; z-index: 2;
      display: flex; align-items: center; gap: 8px;
      width: 134px; padding: 6px 8px;
      background: var(--bg); color: var(--ink);
      border: 1px solid var(--rule);
      border-left: 2px solid var(--layer, var(--rule));
      border-radius: 3px; }
  .p-chip .ico { width: 16px; height: 16px; flex: 0 0 auto;
                 color: var(--layer, var(--dim)); }
  .p-chip .ico svg { width: 16px; height: 16px; display: block; }
  .p-chip span.n { font-size: 0.68rem; line-height: 1.2;
                   white-space: nowrap; }

  /* La bande du dessous : un aperçu par couche, posé sous la colonne qu'il
     illustre. Le bloc d'origine y mettait des `left`/`top` réglés à la
     main ; ici c'est mesuré, donc un libellé plus long ne décale rien. */
  #below { position: relative; width: __W__px; }
  .pk { position: absolute; top: 26px; width: 144px; }
  .pk-table {
      width: 100%; border-collapse: collapse;
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
      font-size: 0.57rem; line-height: 1.35;
      background: var(--panel); border: 1px solid var(--rule); }
  .pk-table th, .pk-table td {
      padding: 2px 6px; text-align: left; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis; }
  .pk-table th { font-weight: 400; color: var(--dim); max-width: 74px; }
  .pk-table td { color: var(--ink); border-left: 1px solid var(--rule);
                 max-width: 70px; }
  .pk-table tr + tr th, .pk-table tr + tr td {
      border-top: 1px solid var(--rule); }
  .pk-code { background: var(--panel); border: 1px solid var(--rule);
             overflow: hidden; }
  .pk-code code {
      display: block; padding: 6px 7px;
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
      font-size: 0.55rem; line-height: 1.5; color: var(--ink);
      white-space: pre; }

  #foot { width: __W__px; margin: 12px auto 0; text-align: right;
          font-size: 0.65rem; color: var(--dim); }
  #foot a { color: inherit; text-decoration: none;
            border-bottom: 1px solid var(--rule); }
  #foot a:hover { color: var(--ink); }

  /* `outline-style` vaut `none` par défaut : sans cette réservation,
     animer outline-color ne rend rien, quelle que soit la couleur.
     outline plutôt que border-color parce qu'il est tracé hors du modèle
     de boîte — il s'allume sans provoquer de recalcul de mise en page. */
  /* Le bout de chaîne ne montre pas un fichier mais un chiffre : la
     plomberie est déjà racontée quatre fois à gauche. */
  .pk-kpi { background: var(--panel); border: 1px solid var(--rule);
            overflow: hidden; }
  .pk-kpi b { display: block; padding: 7px 8px 0;
              font-size: 0.95rem; font-weight: 700; line-height: 1;
              letter-spacing: -0.015em; color: var(--ink);
              font-variant-numeric: tabular-nums; }
  .pk-kpi em { display: block; padding: 3px 8px 8px; font-style: normal;
               font-size: 0.55rem; line-height: 1.3; color: var(--dim); }

  .p-node, .p-chip, .pk { outline: 1.5px solid transparent; }

  @media (prefers-reduced-motion: reduce) {
      #bead { display: none; }
      .p-node { transition: none; }
      .p-node:hover { transform: none; }
  }
</style>

<div id="scroll">
  <div id="scene">
    <svg id="flow" fill="none" aria-hidden="true"></svg>
    <i id="bead" aria-hidden="true"></i>
    <div id="row">
      <div id="srcs">__TILES__</div>
      <div class="gap bus"></div>
      __STORES__
    </div>
    <div id="below">__PEEKS__</div>
    <p id="foot">__FOOT__</p>
  </div>
</div>
"""
_PIPE_JS = r"""
<script>
const SOURCES = __SOURCES__;
const KEYS = __KEYS__;

// Minutage du bloc d'origine, repris tel quel : une vitesse en pixels par
// seconde plutôt qu'une durée fixe, et deux retombées différentes selon
// qu'un bloc garde la trace du passage ou non.
const SPEED = 46;        // px/s
const PEAK = 0.25;       // s — montée du halo
const SETTLE = 0.45;     // s — du pic au tenu (couches)
const DECAY = 0.75;      // s — du pic au neutre (aperçus)
const PAUSE = 0.4;       // s — repos avant la source suivante

const SLOW = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const SVG = "http://www.w3.org/2000/svg";

const scene = document.getElementById("scene");
const flow = document.getElementById("flow");
const bead = document.getElementById("bead");
const below = document.getElementById("below");

const node = id => scene.querySelector('.p-node[data-id="' + id + '"]');
const chip = k => scene.querySelector('.p-chip[data-k="' + k + '"]');
const peek = k => below.querySelector('.pk[data-k="' + k + '"]');

let lap = 0;           // jeton d'annulation : une nouvelle boucle périme l'ancienne
let active = 0;

function mk(cls) {
  const p = document.createElementNS(SVG, "path");
  p.setAttribute("class", cls);
  flow.appendChild(p);
  return p;
}

// Une couche SVG, trois rôles : les chaînes grises du schéma, les traînées
// colorées que la bille laisse derrière elle, et les tiges qui descendent
// vers les aperçus. Tous les `d` sont calculés après mesure.
const chains = [];
const stems = [];
const trails = [];
for (let i = 0; i < SOURCES.length + KEYS.length - 1; i++) {
  chains.push(mk("chain"));
}
for (let i = 0; i < KEYS.length; i++) { stems.push(mk("stem")); }
for (let i = 0; i < KEYS.length; i++) { trails.push(mk("trail")); }

function rgba(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3
    ? h.split("").map(c => c + c).join("") : h, 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + ","
    + (n & 255) + "," + a + ")";
}

function d(a, b) { return "M" + a.x + " " + a.y + "L" + b.x + " " + b.y; }
function len(a, b) { return Math.hypot(b.x - a.x, b.y - a.y); }
function wait(ms) { return new Promise(r => setTimeout(r, ms)); }
function stop(el) { if (el) { el.getAnimations().forEach(a => a.cancel()); } }

// Annuler une animation `fill: forwards` suffit en principe à rendre
// l'élément à sa feuille de style ; on efface aussi les propriétés en
// ligne, pour qu'aucune couleur tenue ne survive d'un tour à l'autre.
function reset(el) {
  if (!el) { return; }
  stop(el);
  el.style.removeProperty("outline-color");
  el.style.removeProperty("box-shadow");
}

let GEO = null;

function measure() {
  const o = scene.getBoundingClientRect();
  const R = el => {
    const b = el.getBoundingClientRect();
    return { l: b.left - o.left, r: b.right - o.left,
             t: b.top - o.top, b: b.bottom - o.top,
             cx: b.left - o.left + b.width / 2,
             cy: b.top - o.top + b.height / 2 };
  };
  const g = { tiles: {}, chips: {}, w: o.width };
  SOURCES.forEach(s => { g.tiles[s.id] = R(node(s.id)); });
  KEYS.forEach(k => { g.chips[k] = R(chip(k)); });
  g.rowCy = g.chips[KEYS[0]].cy;
  flow.setAttribute("width", Math.ceil(o.width));
  flow.setAttribute("height", Math.ceil(o.height));
  return g;
}

// Les aperçus sont centrés sous leur pastille, puis ramenés dans la scène
// s'ils en sortent, et la tige suit : elle part du bas de la pastille et
// arrive au milieu du haut de l'aperçu, droite ou oblique selon le cas.
function place(g) {
  let tall = 0;
  KEYS.forEach((k, i) => {
    const pk = peek(k);
    if (!pk) { return; }
    const w = pk.offsetWidth;
    const x = Math.max(0, Math.min(g.w - w, g.chips[k].cx - w / 2));
    pk.style.left = x + "px";
    tall = Math.max(tall, pk.offsetTop + pk.offsetHeight);
    const b = pk.getBoundingClientRect();
    const o = scene.getBoundingClientRect();
    stems[i].setAttribute("d", d(
      { x: Math.round(g.chips[k].cx), y: Math.round(g.chips[k].b + 1) },
      { x: Math.round(b.left - o.left + b.width / 2),
        y: Math.round(b.top - o.top) }));
  });
  below.style.height = (tall + 2) + "px";
}

function wire(g) {
  let n = 0;
  SOURCES.forEach(s => {
    const k = KEYS[s.enter];
    chains[n++].setAttribute("d", d(
      { x: g.tiles[s.id].r, y: g.tiles[s.id].cy },
      { x: g.chips[k].l, y: g.rowCy }));
  });
  for (let i = 0; i < KEYS.length - 1; i++) {
    chains[n++].setAttribute("d", d(
      { x: g.chips[KEYS[i]].r, y: g.rowCy },
      { x: g.chips[KEYS[i + 1]].l, y: g.rowCy }));
  }
}

// Le contour d'une couche s'allume d'un coup, culmine dans un halo large,
// puis RESTE allumé : c'est la trace du passage, effacée au tour suivant.
function light(el, c) {
  if (!el) { return Promise.resolve(); }
  stop(el);
  const total = PEAK + SETTLE;
  return el.animate([
    { offset: 0, outlineColor: "transparent",
      boxShadow: "0 0 0 0 " + c.gone },
    { offset: PEAK / total, outlineColor: c.flow,
      boxShadow: "0 0 26px 8px " + c.soft },
    { offset: 1, outlineColor: c.flow, boxShadow: "0 0 0 0 " + c.gone },
  ], { duration: total * 1000, fill: "forwards", easing: "linear" })
    .finished.catch(() => {});
}

// Le halo d'un aperçu s'allume, culmine, puis retombe au neutre : un
// aperçu montre ce que la donnée vient de DEVENIR, pas où elle est — il ne
// doit donc pas rester allumé comme une couche.
function flash(el, c) {
  if (!el) { return Promise.resolve(); }
  stop(el);
  const total = PEAK + DECAY;
  return el.animate([
    { offset: 0, outlineColor: "transparent",
      boxShadow: "0 0 0 0 " + c.gone },
    { offset: PEAK / total, outlineColor: c.flow,
      boxShadow: "0 0 34px 10px " + c.soft },
    { offset: 1, outlineColor: "transparent",
      boxShadow: "0 0 0 0 " + c.gone },
  ], { duration: total * 1000, fill: "forwards", easing: "linear" })
    .finished.catch(() => {});
}

// Un segment : la bille et sa traînée, toutes deux réglées sur la distance
// réellement mesurée — une longue diagonale met visiblement plus longtemps
// qu'un saut court.
function travel(trail, a, b, c) {
  const px = len(a, b);
  const dur = Math.max(0.15, px / SPEED);
  trail.setAttribute("d", d(a, b));
  trail.style.stroke = c.flow;
  stop(trail);
  const total = trail.getTotalLength();
  trail.style.strokeDasharray = String(total);
  const t = trail.animate(
    [{ strokeDashoffset: total }, { strokeDashoffset: 0 }],
    { duration: dur * 1000, fill: "forwards", easing: "linear" });
  stop(bead);
  const edge = Math.min(0.4, 0.12 / dur);
  const bb = bead.animate([
    { offset: 0, transform: "translate(" + a.x + "px," + a.y + "px)",
      opacity: 0 },
    { offset: edge, transform: "translate(" + a.x + "px," + a.y + "px)",
      opacity: 1 },
    { offset: 1 - edge, opacity: 1 },
    { offset: 1, transform: "translate(" + b.x + "px," + b.y + "px)",
      opacity: 0 },
  ], { duration: dur * 1000, fill: "forwards", easing: "linear" });
  return Promise.all([t.finished.catch(() => {}),
                      bb.finished.catch(() => {})]);
}

async function play(i) {
  const me = ++lap;
  active = i;
  const s = SOURCES[i];
  const c = { flow: s.color, soft: rgba(s.color, 0.42),
              gone: rgba(s.color, 0) };
  scene.style.setProperty("--flow", s.color);
  SOURCES.forEach(o => node(o.id).classList.toggle("on", o.id === s.id));

  SOURCES.forEach(o => reset(node(o.id)));
  KEYS.forEach(k => { reset(chip(k)); reset(peek(k)); });
  trails.forEach(t => {
    stop(t);
    t.style.strokeDasharray = "0 1";
    t.style.removeProperty("stroke");
  });
  stop(bead);
  bead.style.opacity = 0;

  GEO = measure();
  wire(GEO);
  place(GEO);
  fit();

  const route = KEYS.slice(s.enter);
  const hops = [[{ x: GEO.tiles[s.id].r, y: GEO.tiles[s.id].cy },
                 { x: GEO.chips[route[0]].l, y: GEO.rowCy }]];
  for (let n = 0; n < route.length - 1; n++) {
    hops.push([{ x: GEO.chips[route[n]].r, y: GEO.rowCy },
               { x: GEO.chips[route[n + 1]].l, y: GEO.rowCy }]);
  }

  if (SLOW) {
    node(s.id).style.outlineColor = c.flow;
    route.forEach(k => { chip(k).style.outlineColor = c.flow; });
    hops.forEach((h, n) => {
      trails[n].setAttribute("d", d(h[0], h[1]));
      trails[n].style.stroke = c.flow;
      trails[n].style.strokeDasharray = "none";
    });
    return;
  }

  await light(node(s.id), c);
  if (me !== lap) { return; }
  for (let n = 0; n < hops.length; n++) {
    await travel(trails[n], hops[n][0], hops[n][1], c);
    if (me !== lap) { return; }
    await Promise.all([light(chip(route[n]), c),
                       flash(peek(route[n]), c)]);
    if (me !== lap) { return; }
  }
  await wait(PAUSE * 1000);
  if (me !== lap) { return; }
  play((i + 1) % SOURCES.length);
}

// L'iframe se dimensionne elle-même : la hauteur dépend des aperçus, qui
// dépendent de la largeur disponible. Une valeur en dur rognerait le bloc
// sur un écran étroit et laisserait un trou sur un large.
function fit() {
  try {
    const f = window.frameElement;
    if (!f) { return; }
    const h = Math.ceil(document.body.scrollHeight) + 2;
    f.style.height = h + "px";
    f.setAttribute("height", String(h));
    if (f.parentElement && f.parentElement.style.height) {
      f.parentElement.style.height = h + "px";
    }
  } catch (e) { }
}

SOURCES.forEach((s, i) => {
  node(s.id).addEventListener("click", () => play(i));
});

let timer = null;
window.addEventListener("resize", () => {
  clearTimeout(timer);
  timer = setTimeout(() => {
    GEO = measure();
    wire(GEO);
    place(GEO);
    fit();
  }, 120);
});

if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => play(0));
} else {
  play(0);
}
</script>
"""


def _tiles() -> str:
    out = []
    for s in _sources():
        out.append(
            f'<button type="button" class="p-node" data-id="{s["id"]}"'
            f' style="--src:{s["color"]}">'
            '<span class="p-body"><span class="p-dot"></span>'
            f'<span class="p-ico">{s["glyph"]}</span>'
            f'<span class="p-name">{s["label"]}'
            f'<span class="p-sub">{s["sub"]}</span></span>'
            "</span></button>"
        )
    return "".join(out)


def _stores() -> str:
    out = []
    for i, (name, logo, layers) in enumerate(_cards()):
        if i:
            out.append('<div class="gap"></div>')
        chips = "".join(
            f'<li><span class="p-chip" data-k="{k}"'
            + (f' style="--layer:{tone}"' if tone else "")
            + f'><span class="ico">{icon}</span>'
            f'<span class="n">{label}</span></span></li>'
            for k, label, tone, icon, _ in layers
        )
        out.append(
            f'<div class="store" style="width:{_store_w(len(layers))}px">'
            f'<p class="store-head"><span class="ico">{logo}</span>'
            f"<b>{name}</b></p>"
            f'<ul class="store-list">{chips}</ul></div>'
        )
    return "".join(out)


def _store_w(n: int) -> int:
    """Largeur d'une carte produit : son remplissage, ses pastilles, ses gouttières."""
    return 18 + 134 * n + 24 * (n - 1) + 11


def _peeks() -> str:
    return "".join(
        f'<div class="pk" data-k="{k}">{body}</div>'
        for _, _, layers in _cards()
        for k, _, _, _, body in layers
    )


def _keys() -> list:
    return [k for _, _, layers in _cards() for k, _, _, _, _ in layers]


def _pipeline_component() -> str:
    """Le bloc animé, prêt pour components.html."""
    p = theme.palette()
    srcs = [{"id": s["id"], "color": s["color"], "enter": s["enter"]}
            for s in _sources()]
    return ((_PIPE_HTML + _PIPE_JS)
            .replace("__TILES__", _tiles())
            .replace("__STORES__", _stores())
            .replace("__PEEKS__", _peeks())
            .replace("__FOOT__", _foot())
            .replace("__SOURCES__", json.dumps(srcs))
            .replace("__KEYS__", json.dumps(_keys()))
            .replace("__W__", str(W_SCENE))
            .replace("__INK__", p["TEXT_LIGHT"])
            .replace("__DIM__", p["TEXT_DIM"])
            .replace("__RULE__", p["RULE"])
            .replace("__BG__", p["BG_PAGE"])
            .replace("__PANEL__", p["BG_PANEL"])
            .replace("__CDN__", _CDN))


def _foot() -> str:
    return ('Deux appels publics sur '
            f'<a href="{_CDN}" target="_blank" rel="noopener">'
            "d37ci6vzurychx.cloudfront.net</a>, le CDN de la "
            '<a href="https://www.nyc.gov/site/tlc/about/'
            'tlc-trip-record-data.page" target="_blank" rel="noopener">'
            "NYC Taxi &amp; Limousine Commission</a>.")


def render_pipeline_tab():
    """Une seule chose sur la page : le chemin que prend la donnée.

    La hauteur passée ici n'est qu'un point de départ : le composant la
    corrige lui-même une fois mesuré, puisque la bande d'aperçus dépend de
    la largeur disponible.
    """
    st.header(t("pipe.h.flow"))
    components.html(_pipeline_component(), height=520, scrolling=False)
