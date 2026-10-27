"""
Coeur du dashboard NYC Yellow Taxi — partagé entre les deux entrypoints.

streamlit_dashboard_local.py fournit `load_data()` et délègue tout le rendu à
`run()`. La séparation vient d'une époque où deux entrypoints coexistaient (un
sur Snowflake, un sur DuckDB) ; elle reste utile — la couche d'accès aux données
est isolée du rendu.

Structure de l'interface :

    [sidebar]          À propos | Pipeline de données | Dashboard (par défaut)
     Section
      Combien ?
      Quand ?
      Où ?
      Comment ?

Les trois sections du dashboard sont dans la sidebar et non dans un second rang
d'onglets : Streamlit ne sait pas rendre `st.tabs` verticalement, et empiler
deux rangées horizontales alourdissait le haut de page. La sidebar est écrite
depuis `run()`, donc avant les onglets — c'est un contrôle global, pas un
élément d'un onglet (`st.tabs` exécute de toute façon les trois corps à chaque
rerun, l'état actif n'est pas connu côté Python).
"""

import pathlib

import pandas as pd
import streamlit as st

import nyc_content
import nyc_lang
import nyc_cockpit
import nyc_theme
import nyc_views

# Le logo sert à la fois de favicon et de marque dans la barre. Chemin absolu
# dérivé du module : l'application est lancée depuis des répertoires de travail
# différents en local et dans l'image Docker.
LOGO_PATH = str(pathlib.Path(__file__).resolve().parent / "assets" / "logo-icon.svg")

PAGE_CONFIG = dict(
    page_title="NYC Yellow Taxi",
    page_icon=LOGO_PATH,
    layout="wide",
    # Un entier à la place de "expanded" fixe la largeur de la barre en
    # pixels (Streamlit 1.66, borné à 200-600 — 200 est le minimum) et met
    # l'état sur "auto" :
    # ouverte sur un écran large, repliée sur un écran étroit. Le visiteur
    # peut toujours la tirer par son bord ; sa largeur est alors retenue.
    initial_sidebar_state=200,
)

# Les vues sont des clés, pas des intitulés : elles servent à la fois de
# valeur à `?vue=` et de clé de traduction (`nav.` + la clé). Un lien partagé
# reste donc valide quelle que soit la langue de celui qui l'ouvre, et le code
# ne compare jamais du texte affiché.
PAGE_COCKPIT  = "cockpit"
PAGE_ABOUT    = "about"
PAGE_PIPELINE = "pipeline"


# Le dashboard est découpé en questions plutôt qu'en catégories : court,
# mémorisable, et traduisible sans perte (How much / When / Where / How).
SEC_HOW_MUCH = "how_much"
SEC_WHEN     = "when"
SEC_WHERE    = "where"
SEC_HOW      = "how"
SECTIONS = [SEC_HOW_MUCH, SEC_WHEN, SEC_WHERE, SEC_HOW]

# Une seule liste pour la barre : les sections du dashboard en tête, les pages
# de contexte en dessous. Le CSS distingue les deux groupes en visant les
# options au-delà de la quatrième — d'où N_PRIMARY, qui doit rester accordé.
VIEWS = [PAGE_COCKPIT] + SECTIONS + [PAGE_ABOUT, PAGE_PIPELINE]
N_PRIMARY = len(SECTIONS) + 1
# "Quand ?" est la section la plus riche : c'est elle qui ouvre.
DEFAULT_SECTION = SEC_WHEN

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def run(load_data):
    """Point d'entrée unique. `load_data` est la fonction (mise en cache par
    l'appelant) qui renvoie le tuple (daily, hourly, zones, profile)."""
    # Pas de st.logo() : il ne sait poser la marque qu'au-dessus de la barre
    # latérale, ou dans le bandeau une fois celle-ci repliée. Le logo est donc
    # dessiné par nyc_theme, à gauche du titre dans le bandeau, où il reste
    # dans les deux états. LOGO_PATH sert encore d'icône d'onglet.
    nyc_theme.inject_css()
    # Hors de st.sidebar : le CSS épingle le groupe dans la bannière en
    # position fixe, et une barre repliée est déplacée par un `transform`, qui
    # ferait alors office de bloc conteneur et emporterait le groupe hors de
    # l'écran avec elle.
    nyc_theme.band()

    daily, hourly, zones, profile, congestion, split, payment = _load(load_data)
    view = _sidebar()

    if view == PAGE_COCKPIT:
        _scope_strip(daily)
        nyc_cockpit.render(daily, split, zones, profile)
    elif view == PAGE_ABOUT:
        st.title("NYC Yellow Taxi")
        nyc_content.render_about_tab()
    elif view == PAGE_PIPELINE:
        st.title("NYC Yellow Taxi")
        nyc_content.render_pipeline_tab()
    else:
        _scope_strip(daily)
        if view == SEC_HOW_MUCH:
            nyc_views.how_much(daily, split, _cross("od_borough"), congestion)
        elif view == SEC_WHEN:
            nyc_views.when(daily, _cross("hour_weekday"))
        elif view == SEC_WHERE:
            nyc_views.where(zones, _cross("od_borough"),
                            _cross("dropoff_zones"), _cross("airport_monthly"))
        else:
            nyc_views.how(profile, _cross("distance_bins"), payment,
                          _cross("passengers"))

    st.markdown("---")
    st.caption("Source : NYC Taxi & Limousine Commission (TLC) — Yellow Taxi Trip Records")


def _load(load_data):
    with st.spinner("Chargement des données..."):
        try:
            daily, hourly, zones, profile, congestion, split, payment = load_data()
            daily["PICKUP_DATE"] = pd.to_datetime(daily["PICKUP_DATE"], errors="coerce")
            daily = daily.dropna(subset=["PICKUP_DATE"])
            # Filtre défensif : on coupe à fin oct. 2025 même si le cache est ancien
            daily = daily[daily["PICKUP_DATE"] <= pd.Timestamp("2025-10-31")]
            # Exclure les jours avec données incomplètes (< 30 % de la médiane)
            _med = daily["TOTAL_TRIPS"].median()  # type: ignore
            daily = daily[daily["TOTAL_TRIPS"] >= _med * 0.30]
            if daily.empty:  # type: ignore
                st.warning("Aucune donnée valide dans daily_summary.")
                st.stop()
        except Exception as e:
            st.error(f"Erreur de chargement : {e}")
            st.stop()
    return daily, hourly, zones, profile, congestion, split, payment


# Tables croisées : construites hors ligne par scripts/G_build_aggregates.py,
# elles ne sont pas dans le tuple de load_data(). Celui-ci sait reconstruire
# ses sept tables depuis les Parquet bruts si elles manquent ; les croisements
# demandent une relecture complète des 74 M de lignes et ne peuvent pas être
# calculés dans une requête web. D'où une lecture séparée, qui rend None quand
# le fichier n'existe pas — la section concernée l'annonce alors elle-même.
AGG_DIR = pathlib.Path(__file__).resolve().parent / "data" / "aggregates"


@st.cache_data(ttl=3600)
def _cross(name):
    path = AGG_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def _sidebar():
    """Les quatre sections du dashboard, puis les deux pages de contexte.

    Un seul `st.radio` et non deux listes empilées : deux widgets porteraient
    deux sélections indépendantes, et il faudrait arbitrer laquelle a changé
    en dernier. Ici l'état est unique, et la hiérarchie est affaire de CSS.
    """
    with st.sidebar:
        view = st.radio(
            "Vue", VIEWS, index=_index(VIEWS, "vue", DEFAULT_SECTION),
            format_func=lambda key: nyc_lang.t("nav." + key),
            label_visibility="collapsed", key="nav_view",
        )
        # La bascule de thème recharge la page : sans ce paramètre d'URL,
        # l'état serait perdu et l'on repartirait sur la vue par défaut.
        # Bonus : les vues deviennent partageables.
        st.query_params["vue"] = view
    return view


def _scope_strip(daily):
    """Trois nombres qui situent les données, sur une ligne, sous le titre."""
    st.markdown(
        nyc_theme._html(
            f"<div class='scope-strip'>{nyc_theme.fmt_int(daily['TOTAL_TRIPS'].sum())}"
            f" courses · {len(daily)} jours · "
            f"{daily['PICKUP_DATE'].min():%b %Y} – {daily['PICKUP_DATE'].max():%b %Y}</div>"
        ),
        unsafe_allow_html=True,
    )


def _index(options, param, default):
    """Position de l'option nommée dans l'URL, sinon celle par défaut."""
    value = st.query_params.get(param)
    return options.index(value) if value in options else options.index(default)


