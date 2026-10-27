"""Le cockpit — une page, une horloge, cinq blocs qui répondent ensemble.

Ce n'est pas une page de graphiques posés côte à côte : c'est une seule
machine. Un cycle tourne — la journée, la semaine ou l'année type — et tout
ce qui est affiché se recompose au même pas. La carte, le ruban, les flux et
les quatre nombres montrent la même heure de la même journée, sinon ils ne
disent rien.

Trois bandes, et les commandes au milieu :

    ┌───────────────┬───────────────────────────────┐
    │ 4 nombres     │ ruban qui défile              │
    ├───────────────┴───────────────────────────────┤
    │ cycle · mesure · pas courant · lecture        │
    ├───────────────────────────────┬───────────────┤
    │ carte à pics                  │ d'où l'on     │
    │                               │ part, où l'on │
    │                               │ arrive        │
    └───────────────────────────────┴───────────────┘

Les commandes sont au centre et non en tête : elles séparent ce qui résume
— les nombres, le ruban — de ce qui situe — la carte, les trajets. Elles
tombent ainsi à portée des deux, au lieu de surplomber l'un et d'abandonner
l'autre en bas de page.

Streamlit ne connaît pas les grilles : `st.columns` prend des poids, et
l'empilement vertical vient de l'ordre d'écriture. Chaque bande est donc une
rangée de colonnes, et les hauteurs sont fixées en pixels pour que deux
blocs voisins s'alignent.

Chaque bloc porte son titre à l'intérieur de son iframe, en surimpression.
Un titre posé par Streamlit au-dessus du bloc ajouterait une marge propre à
chaque bloc, et les deux colonnes cesseraient de s'aligner.

Tout ce qui bouge est dans nyc_live : cinq iframes qui se parlent par
BroadcastChannel, sans réveiller Python.
"""

import streamlit as st

import nyc_live

# Hauteurs des trois bandes, en pixels. Les composants vivent dans des
# iframes de taille fixe : sans ces valeurs, deux blocs voisins ne s'alignent
# pas.
H_TOP = 170
H_CONTROLS = 54
H_BOTTOM = 440


def render():
    if not nyc_live.available():
        st.info(
            "Les agrégats du cockpit sont absents. Les construire :\n\n"
            "```\npython scripts/H_build_geo.py\n"
            "python scripts/J_build_cycles.py\n```"
        )
        return

    numbers, ribbon = st.columns([2, 3], gap="medium")
    with numbers:
        nyc_live.kpis(height=H_TOP)
    with ribbon:
        nyc_live.profile(
            height=H_TOP,
            title="Le volume de la ville, pas par pas",
            hint="Le présent est sur le trait ; le passé file à gauche.",
        )

    nyc_live.controls(height=H_CONTROLS)

    city, trips = st.columns([3, 2], gap="medium")
    with city:
        nyc_live.city_map(
            height=H_BOTTOM,
            title="New York au pas de l'horloge",
            hint="Hauteur et couleur d'une zone : la mesure choisie, au pas "
                 "affiché.",
        )
    with trips:
        nyc_live.flows(
            height=H_BOTTOM,
            title="D'où l'on part, où l'on arrive",
            hint="Manhattan coupé en quatre, du sud au nord : à l'échelle de "
                 "l'arrondissement, 83 % des courses ne bougeaient pas. "
                 "Trajets internes à un secteur écartés.",
        )
