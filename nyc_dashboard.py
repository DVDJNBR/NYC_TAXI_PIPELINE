"""
Coeur du dashboard NYC Yellow Taxi — partagé entre les deux entrypoints.

streamlit_dashboard_local.py fournit `load_data()` et délègue tout le rendu à
`run()`. La séparation vient d'une époque où deux entrypoints coexistaient (un
sur Snowflake, un sur DuckDB) ; elle reste utile — la couche d'accès aux données
est isolée du rendu.

Structure de l'interface :

    [bandeau]   Dashboard · À propos · Pipeline de données | logo + titre |
                thème | langue

Le dashboard est une grille unique — le cockpit — et non plus quatre sections
à parcourir. Une barre latérale pour trois entrées prenait une colonne entière
pour ne rien dire : les onglets sont passés dans le bandeau, où ils tiennent
sur une ligne et rendent sa largeur à la grille.
"""

import pathlib

import pandas as pd
import streamlit as st

import nyc_content
import nyc_lang
import nyc_cockpit
import nyc_theme

# Le logo sert à la fois de favicon et de marque dans la barre. Chemin absolu
# dérivé du module : l'application est lancée depuis des répertoires de travail
# différents en local et dans l'image Docker.
LOGO_PATH = str(pathlib.Path(__file__).resolve().parent / "assets" / "logo-icon.svg")

PAGE_CONFIG = dict(
    page_title="NYC Yellow Taxi",
    page_icon=LOGO_PATH,
    layout="wide",
    # Plus de barre latérale : rien n'est écrit dans st.sidebar, et Streamlit
    # ne rend ni la colonne ni son bouton de dépli.
    initial_sidebar_state="collapsed",
)

# Les vues sont des clés, pas des intitulés : elles servent à la fois de
# valeur à `?vue=` et de clé de traduction (`nav.` + la clé). Un lien partagé
# reste donc valide quelle que soit la langue de celui qui l'ouvre, et le code
# ne compare jamais du texte affiché.
PAGE_COCKPIT  = "cockpit"
PAGE_ABOUT    = "about"
PAGE_PIPELINE = "pipeline"


# Une seule liste pour la barre : le cockpit, puis les pages de contexte. Le
# CSS distingue les deux groupes en visant les options au-delà de la première
# — d'où N_PRIMARY, qui doit rester accordé.
#
# Les quatre sections ont quitté la navigation : le cockpit les a remplacées.
# Elles posaient seize figures côte à côte, chacune répondant à sa question
# dans son coin ; le cockpit en pose quatre qui répondent ensemble, au même
# pas de la même journée. Leur code reste dans nyc_views, il n'est plus
# atteignable depuis la barre.
VIEWS = [PAGE_COCKPIT, PAGE_ABOUT, PAGE_PIPELINE]
N_PRIMARY = 1
DEFAULT_SECTION = PAGE_COCKPIT

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
    # Les onglets sont rendus par band(), dans le groupe en position fixe de
    # la bannière : un widget écrit ailleurs atterrirait dans le corps de la
    # page. D'où la navigation passée en fonction plutôt qu'appelée ici.
    nyc_theme.band(nav=_nav)

    daily, hourly, zones, profile, congestion, split, payment = _load(load_data)
    view = st.session_state.get("nav_view", DEFAULT_SECTION)

    if view == PAGE_ABOUT:
        st.title("NYC Yellow Taxi")
        nyc_content.render_about_tab()
    elif view == PAGE_PIPELINE:
        st.title("NYC Yellow Taxi")
        nyc_content.render_pipeline_tab()
    else:
        nyc_cockpit.render()

    _footer(daily)


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


def _nav():
    """Les trois onglets, rendus dans le bandeau par nyc_theme.band().

    Un `st.radio` et non trois boutons : l'état est unique et porté par le
    widget, là où trois boutons demanderaient d'arbitrer lequel a été cliqué
    en dernier. Les pastilles sont masquées en CSS — une case cochée est un
    formulaire, pas une barre d'onglets.
    """
    view = st.radio(
        "Vue", VIEWS, index=_index(VIEWS, "vue", DEFAULT_SECTION),
        format_func=lambda key: nyc_lang.t("nav." + key),
        horizontal=True, label_visibility="collapsed", key="nav_view",
    )
    # La bascule de thème recharge la page : sans ce paramètre d'URL, l'état
    # serait perdu et l'on repartirait sur la vue par défaut. Bonus : les vues
    # deviennent partageables.
    st.query_params["vue"] = view
    return view


_TLC_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"


def _footer(daily):
    """Provenance et amplitude, sur une ligne, en bas à droite.

    Ces nombres situent les données, ils ne les racontent pas : en tête de
    page ils prenaient de la hauteur au-dessus de ce qu'ils décrivent, et le
    cockpit perdait d'autant. En pied, ils se lisent quand on les cherche.

    Ils voisinent avec la source parce qu'ils disent la même chose : ce que
    couvre le jeu de données, et d'où il vient.
    """
    st.markdown("---")
    st.markdown(
        nyc_theme._html(
            "<div class='page-foot'>"
            f"<a href='{_TLC_URL}' target='_blank' rel='noopener'>"
            "NYC Taxi &amp; Limousine Commission</a> · Yellow Taxi Trip "
            f"Records · {nyc_theme.fmt_int(daily['TOTAL_TRIPS'].sum())}"
            f" courses · {len(daily)} jours · "
            f"{daily['PICKUP_DATE'].min():%b %Y} – "
            f"{daily['PICKUP_DATE'].max():%b %Y}</div>"
        ),
        unsafe_allow_html=True,
    )


def _index(options, param, default):
    """Position de l'option nommée dans l'URL, sinon celle par défaut."""
    value = st.query_params.get(param)
    return options.index(value) if value in options else options.index(default)


