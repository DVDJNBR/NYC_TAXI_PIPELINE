"""Le cockpit — une page, une horloge, cinq blocs qui répondent ensemble.

Ce n'est pas une page de graphiques posés côte à côte : c'est une seule
machine. Un cycle tourne — la journée, la semaine ou l'année type — et tout
ce qui est affiché se recompose au même pas. La carte, le ruban, les flux et
les deux cadrans montrent la même heure de la même journée, sinon ils ne
disent rien.

Trois bandes, et les commandes au milieu :

    ┌───────────────┬───────────────────────────────┐
    │ d'où l'on     │ carte à pics                  │
    │ part, où l'on │                               │
    │ arrive        │                               │
    ├───────────────┴───────────────────────────────┤
    │ cycle · mesure · pas courant · lecture        │
    ├───────────────┬───────────────────────────────┤
    │ passagers     │                               │
    ├───────────────┤ ruban qui défile              │
    │ vitesse       │                               │
    └───────────────┴───────────────────────────────┘

Les commandes sont au centre et non en tête : elles séparent ce qui situe
— la carte, les trajets — de ce qui résume — les nombres, le ruban. Elles
tombent ainsi à portée des deux, au lieu de surplomber l'un et d'abandonner
l'autre en bas de page.

Ce qui situe vient d'abord, et ce qui résume ensuite : on arrive sur la
ville avant d'arriver sur ses nombres. Le ruban est en bas parce que c'est
lui qu'on lit en dernier, une fois qu'on sait où l'on est.

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
# La bande des plans porte les deux dessins qui situent : les échanges entre
# secteurs et la carte à pics.
# La bande des cadrans porte l'habitacle : le volant des passagers en haut, le
# compteur de vitesse et le taximètre en bas. Deux cases empilées et non côte
# à côte — chacune prend ainsi toute la largeur de la colonne. Sa hauteur est
# celle du ruban d'à côté, qui la partage : c'est lui qui la fixe, pas les
# cases.
# 262 et non 236 : le volant mesure 96 px de jante, et le titre comme le
# sous-titre en prennent treize chacun. À 236 les deux rangs valaient 114 px
# et la jante était rognée. Le ruban prend les vingt-six pixels de plus sans
# rien y perdre — il n'a pas de hauteur propre, il s'étire.
H_PLANS = 440
H_CONTROLS = 70
H_CADRANS = 262


def render():
    if not nyc_live.available():
        st.info(
            "Les agrégats du cockpit sont absents. Les construire :\n\n"
            "```\npython scripts/H_build_geo.py\n"
            "python scripts/J_build_cycles.py\n```"
        )
        return

    # Les deux blocs du haut portent un titre et la légende des secteurs,
    # rien d'autre. Une phrase d'explication se replie sur deux lignes d'un
    # côté et trois de l'autre selon la largeur : les deux dessins ne
    # commençaient alors pas à la même hauteur, et la paire se décrochait.
    # Même titre sur une ligne, même légende en bas, donc même surface
    # utile — c'est ce qui les aligne.
    #
    # Les échanges à gauche, la carte à droite : on lit d'abord les secteurs
    # qui s'échangent les courses, puis où ces courses tombent dans la ville.
    # La carte garde la plus grande part — c'est elle qui porte la géographie,
    # et les échanges tiennent dans une colonne étroite.
    trips, city = st.columns([2, 3], gap="medium")
    with trips:
        nyc_live.flows(
            height=H_PLANS,
            title="Les échanges de courses entre secteurs",
            legend=True,
        )
    with city:
        nyc_live.taxi_city(
            height=H_PLANS,
            title="Les flux de courses entre zones new-yorkaises",
            legend=True,
        )

    nyc_live.controls(height=H_CONTROLS)

    # Moitié-moitié, et non deux cinquièmes : les deux dessins portent leur
    # légende à côté d'eux, et à deux cinquièmes elle passait dessous. Le
    # ruban, lui, lit ses vingt-quatre barres aussi bien sur la moitié de la
    # page.
    numbers, ribbon = st.columns([1, 1], gap="medium")
    with numbers:
        nyc_live.kpis(height=H_CADRANS)
    with ribbon:
        nyc_live.profile(
            height=H_CADRANS,
            # Ce titre ne vit qu'une image : le composant le réécrit dès
            # son premier état, avec la mesure allumée. Il reste ici parce
            # qu'un titre vide fait retirer la barre de titre, et le dessin
            # monterait alors de huit pixels avant de redescendre.
            title="Volume de courses par heure",
            hint="Le présent est sur le trait ; le passé file à gauche.",
        )
