"""Le cockpit — une page, une horloge, quatre blocs qui répondent ensemble.

Ce n'est pas une page de graphiques posés côte à côte : c'est une seule
machine. Un cycle tourne — la journée, la semaine ou l'année type — et tout
ce qui est affiché se recompose au même pas. La carte, le ruban et les flux
montrent la même minute de la même journée, sinon ils ne disent rien.

La grille suit le découpage demandé, huit colonnes sur six rangées :

    ┌──────────────────────────────────────────────┐
    │ boutons-panneaux · horloge                   │  commandes
    ├───────────────────────┬──────────────────────┤
    │ ruban qui défile      │ flux entre           │
    │                       │ arrondissements      │
    ├───────────────────────┴───────────┬──────────┤
    │ carte à pics              5 col   │ classe-  │
    │                                   │ ment     │
    └───────────────────────────────────┴──────────┘

Streamlit ne connaît pas les grilles : `st.columns` prend des poids, et
l'empilement vertical vient de l'ordre d'écriture. Les hauteurs sont donc
fixées à la main, en pixels, pour que les blocs d'une même rangée
s'alignent.

Tout ce qui bouge est dans nyc_live : quatre iframes qui se parlent par
BroadcastChannel, sans réveiller Python. Ce module ne fait que poser les
blocs, et dessiner ce qui ne bouge pas.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nyc_live
import nyc_theme
import nyc_zones

# Hauteurs des rangées, en pixels. Les composants vivent dans des iframes de
# taille fixe : sans ces valeurs, deux blocs voisins ne s'alignent pas.
H_CONTROLS = 86
H_MIDDLE = 300
H_MAP = 470


def render(daily, split, zones, profile):
    if not nyc_live.available():
        st.info(
            "Les agrégats du cockpit sont absents. Les construire :\n\n"
            "```\npython scripts/H_build_geo.py\n"
            "python scripts/J_build_cycles.py\n```"
        )
        return

    _kpis(daily, split, profile)
    nyc_live.controls(height=H_CONTROLS)

    left, right = st.columns([4, 4], gap="medium")
    with left:
        _caption("Le volume de la ville, pas par pas",
                 "Le présent est sur le trait ; le passé file à gauche.")
        nyc_live.profile(height=H_MIDDLE)
    with right:
        _caption("D'où l'on part, où l'on arrive",
                 "Courses internes à un arrondissement écartées : "
                 "à 87 % du total, elles écrasaient tout le reste.")
        nyc_live.flows(height=H_MIDDLE)

    bottom, side = st.columns([5, 3], gap="medium")
    with bottom:
        _caption("New York au pas de l'horloge",
                 "Hauteur et couleur d'une zone : la mesure choisie, "
                 "au pas affiché.")
        nyc_live.city_map(height=H_MAP)
    with side:
        _caption("Les zones qui portent le volume",
                 "Sur l'ensemble de la période, toutes heures confondues.")
        _zone_rank(zones, height=H_MAP)


def _caption(title, note):
    p = nyc_theme.palette()
    st.markdown(
        f"<div style='margin:4px 0 6px;'>"
        f"<div style='font-size:0.95rem; font-weight:600;'>{title}</div>"
        f"<div style='font-size:0.78rem; color:{p['TEXT_DIM']}; "
        f"line-height:1.45;'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def _kpis(daily, split, profile):
    """Quatre nombres qui situent la période, au-dessus de tout le reste.

    Ils ne suivent pas l'horloge : ce sont les bornes de l'étude, pas une
    mesure du pas courant. Les faire bouger laisserait croire qu'une heure
    de la journée type pèse 2,08 milliards de dollars.
    """
    r = split.iloc[0]
    pr = profile.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        nyc_theme.card("Courses analysées",
                       nyc_theme.fmt_int(daily["TOTAL_TRIPS"].sum()),
                       f"{len(daily)} jours, janv. 2024 – sept. 2025")
    with c2:
        nyc_theme.card("Encaissé", f"${r['TOTAL'] / 1e9:.2f}B",
                       f"${r['TOTAL'] / r['TOTAL_TRIPS']:.2f} par course")
    with c3:
        nyc_theme.card("Zones desservies", "231",
                       "sur les 263 du référentiel TLC")
    with c4:
        nyc_theme.card("Courses via un aéroport",
                       f"{pr['PCT_AEROPORT_JFK'] + pr['PCT_AEROPORT_LGA']:.0f} %",
                       "JFK et LaGuardia, départ ou arrivée")


def _zone_rank(zones, height):
    """Classement des zones de départ — le seul bloc qui ne bouge pas.

    Il sert de repère : pendant que la carte change de forme à chaque pas,
    celui-ci rappelle qui domine sur l'ensemble de la période.
    """
    p = nyc_theme.palette()
    z = zones.copy()
    z["ZONE_NAME"] = z["ZONE_ID"].map(nyc_zones.names())
    z["BOROUGH"] = z["ZONE_ID"].map(nyc_zones.boroughs())
    z = z.dropna(subset=["ZONE_NAME"]).nlargest(14, "TOTAL_TRIPS")
    z = z.sort_values("TOTAL_TRIPS")

    fig = go.Figure(go.Bar(
        x=z["TOTAL_TRIPS"], y=z["ZONE_NAME"], orientation="h",
        marker=dict(color=[nyc_zones.color(b) for b in z["BOROUGH"]]),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} départs<extra></extra>",
    ))
    nyc_theme.style_fig(fig)
    fig.update_layout(
        height=height, showlegend=False,
        margin=dict(t=6, b=6, l=0, r=8),
        xaxis=dict(title="", showticklabels=False, showgrid=False),
        yaxis=dict(title="", tickfont=dict(size=10, color=p["TEXT_DIM"])),
    )
    st.plotly_chart(fig, use_container_width=True)
