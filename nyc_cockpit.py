"""Le cockpit — une page, une horloge, cinq blocs qui répondent ensemble.

Ce n'est pas une page de graphiques posés côte à côte : c'est une seule
machine. Un cycle tourne — la journée, la semaine ou l'année type — et tout
ce qui est affiché se recompose au même pas. La carte, le ruban, les flux et
les quatre nombres montrent la même heure de la même journée, sinon ils ne
disent rien.

La grille fait cinq colonnes sur quatre rangées, et la carte occupe trois
colonnes sur trois rangées :

    ┌───────────────┬───────────────────────────────┐
    │ 4 nombres     │ ruban qui défile              │  rangée 1
    ├───────────────┼───────────────────────────────┤
    │ commandes     │                               │  rangée 2
    ├───────────────┤  carte à pics          3 x 3  │
    │ flux entre    │                               │  rangée 3
    │ arrondis-     │                               │
    │ sements  2x2  │                               │  rangée 4
    └───────────────┴───────────────────────────────┘

Streamlit ne connaît pas les grilles : `st.columns` prend des poids, et
l'empilement vertical vient de l'ordre d'écriture. Une cellule ne peut donc
pas s'étendre sur plusieurs rangées. Deux colonnes de poids 2 et 3 donnent
les cinq colonnes ; les rangées sont reconstituées en empilant des blocs dont
la hauteur est un multiple exact de l'unité de rangée, l'écart inter-blocs
compris. D'où le calcul explicite de `_ROWS` plus bas : c'est ce qui fait que
le bas de la carte et le bas des flux tombent sur la même ligne.

Chaque bloc porte son titre à l'intérieur de son iframe, en surimpression.
Un titre posé par Streamlit au-dessus du bloc ajouterait une marge propre à
chaque bloc, et les deux colonnes cesseraient de s'aligner.

Tout ce qui bouge est dans nyc_live : cinq iframes qui se parlent par
BroadcastChannel, sans réveiller Python.
"""

import streamlit as st

import nyc_live

# Unité de rangée, en pixels, et écart vertical entre deux blocs empilés —
# celui que Streamlit pose entre deux éléments d'une colonne.
ROW = 164
GAP = 16


def _rows(n: int) -> int:
    """Hauteur d'un bloc couvrant `n` rangées, écarts internes compris."""
    return ROW * n + GAP * (n - 1)


def render():
    if not nyc_live.available():
        st.info(
            "Les agrégats du cockpit sont absents. Les construire :\n\n"
            "```\npython scripts/H_build_geo.py\n"
            "python scripts/J_build_cycles.py\n```"
        )
        return

    left, right = st.columns([2, 3], gap="medium")

    with left:
        nyc_live.kpis(height=_rows(1))
        nyc_live.controls(height=_rows(1))
        nyc_live.flows(
            height=_rows(2),
            title="D'où l'on part, où l'on arrive",
            hint="Courses internes à un arrondissement écartées : à 87 % du "
                 "total, elles écrasaient tout le reste.",
        )

    with right:
        nyc_live.profile(
            height=_rows(1),
            title="Le volume de la ville, pas par pas",
            hint="Le présent est sur le trait ; le passé file à gauche.",
        )
        nyc_live.city_map(
            # Trois rangées : les mêmes que celles qu'occupent, à gauche, les
            # commandes puis les flux — 164 + 16 + 344 font bien 524.
            height=_rows(3),
            title="New York au pas de l'horloge",
            hint="Hauteur et couleur d'une zone : la mesure choisie, au pas "
                 "affiché.",
        )
