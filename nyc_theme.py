"""
Thème visuel partagé — dashboards NYC Yellow Taxi.

Parti pris : sobriété. Un seul accent (le jaune taxi), un neutre, et deux
couleurs strictement fonctionnelles (meilleur / pire). Pas de dégradé, pas
d'ombre portée, pas d'icône décorative — les angles sont quasi droits.

Clair / sombre
--------------
Les deux palettes vivent dans `.streamlit/config.toml`, sous les tables
`[theme.light]` et `[theme.dark]`. Streamlit applique le bon jeu côté
navigateur, sans rerun : widgets natifs, bordures, graphiques natifs suivent
donc le thème tout seuls. Faute de `base` dans le fichier, le thème par défaut
est celui du système du visiteur.

Ces tables exigent Streamlit >= 1.66 : les versions antérieures les rejettent
(« "theme.dark" is not a valid config option ») et retombent sur leurs propres
préréglages, ce qui fait diverger les widgets natifs de notre palette.

Ce module relit ce même fichier (`_load`) et en déduit sa palette Python, via
`_MAP` qui traduit les noms Streamlit en noms métier. Les couleurs ne sont donc
écrites qu'à un seul endroit. Seul GRID_LINE est dérivé : Streamlit n'a pas
d'option pour la grille des graphiques, et il la faut semi-transparente.

Le Python reste nécessaire parce que deux choses ne peuvent pas suivre le thème
sans re-rendu : notre HTML sur mesure (cartes, rails du pipeline) et les
couleurs Plotly, calculées côté serveur. Streamlit 1.50 n'expose aucune
variable CSS ni attribut de thème dans le DOM — vérifié : zéro custom property
dans les 7 feuilles de style de la page. D'où `band()`, qui recharge.

Les noms historiques (DUSK_ORANGE, NIGHT_PURPLE…) sont conservés pour ne pas
casser les appels existants, mais plusieurs pointent sur le même neutre : les
cartes d'indicateurs ne doivent plus former un arc-en-ciel. De même TEXT_LIGHT
désigne le texte principal, clair sur fond sombre et sombre sur fond clair.

Les couleurs ne sont pas des constantes mais des attributs résolus à chaque
accès, via le `__getattr__` de module (PEP 562) défini en bas de fichier.
`nyc_theme.ACCENT` s'écrit pareil côté appelants mais renvoie la valeur du
thème actif. Corollaire : à l'intérieur de ce fichier il faut passer par
`palette()`, car une lecture de variable globale ne déclenche pas
`__getattr__`.
"""
import functools
import base64
import pathlib
import random
import re
import tomllib

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import nyc_lang

_CONFIG = pathlib.Path(__file__).resolve().parent / ".streamlit" / "config.toml"

# Le logo est posé en image de fond du bandeau, donc encodé en data URI.
# st.logo() ne sait le placer qu'au-dessus de la barre latérale, ou dans le
# bandeau quand elle est repliée ; ici il doit rester à côté du titre dans
# les deux cas.
_LOGO = base64.b64encode(
    (pathlib.Path(__file__).resolve().parent / "assets" / "logo-icon.svg").read_bytes()
).decode()
_LOGO_URI = f"url('data:image/svg+xml;base64,{_LOGO}')"

# Nom métier -> option Streamlit. NEUTRAL et TEXT_DIM tombent volontairement
# sur la même valeur : le neutre de la palette *est* la couleur du texte
# secondaire.
_MAP = {
    "BG_PAGE": "backgroundColor",
    "BG_PANEL": "secondaryBackgroundColor",
    "BG_PANEL_2": "codeBackgroundColor",
    "ACCENT": "primaryColor",
    "NEUTRAL": "grayColor",
    "TEXT_DIM": "grayColor",
    "TEXT_LIGHT": "textColor",
    "BETTER": "greenColor",
    "WORSE": "redColor",
    "RULE": "borderColor",
    "BAR_MUTED": "grayBackgroundColor",
}

# Pas d'équivalent natif : la grille doit rester translucide pour ne pas
# trancher sur le fond, ce qu'un hexadécimal ne permet pas.
# Jonc chromé serti en haut et en bas de la plaque, à la place d'un filet
# blanc dessiné : sur un vrai panneau, ces bords sont du métal replié.
_METAL_BAND = ("linear-gradient(to bottom, #8d8d97 0%, #e8e8f0 42%,"
               " #a2a2ac 62%, #6e6e78 100%)")

# Aplat vert : léger dégradé en diagonale plutôt qu'une couleur plate. Une
# tôle peinte prend la lumière de biais ; un aplat uniforme fait de l'aplat
# vectoriel, pas du métal.
_SIGN_FACE = ("linear-gradient(152deg, #0a7049 0%, #00603f 46%, #004c32 100%)")

# Panneau de rue new-yorkais : aplat vert, texte blanc en capitales, filet
# blanc en retrait du bord. Le vert n'entre pas dans la palette — c'est une
# matière, employée à un seul endroit, comme l'est le chrome du poteau.
_SIGN_GREEN = "#00603f"
_SIGN_INK = "#f4f6f4"
# Le glyphe est sorti en constante : une f-string Python n'accepte pas
# d'antislash dans sa partie expression, et donc pas l'échappement CSS.
_SIGN_ARROW_MASK = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'"
    " viewBox='0 0 24 12'%3E%3Cpath d='M0 4.4h13.3V0.6L23.4 6l-10.1 5.4V7.6H0z'"
    " fill='%23000'/%3E%3C/svg%3E\") center / contain no-repeat"
)

# Cerclage chromé de la ligne active : dégradé vertical, bord sombre puis
# arête claire, comme un métal poli qui prend la lumière du ciel.
#
# L'arête reste en deçà du blanc. Un métal brossé réfléchit un ciel gris, pas
# une source ponctuelle ; poussée trop haut, la bande claire vire au néon et
# le cerclage cesse de se lire comme une pièce.
_METAL = {
    "dark": ("linear-gradient(to bottom, #4e4e57 0%, #9b9ba6 44%, "
             "#787882 56%, #45454d 100%)"),
    "light": ("linear-gradient(to bottom, #86868f 0%, #cbcbd3 44%, "
              "#a3a3ac 56%, #767680 100%)"),
}

# Poussières en suspension : c'est ce qui rend un faisceau visible dans l'air.
#
# Une couche de fond par point, chacune avec sa trajectoire propre. Une seule
# couche déplacée en bloc ferait glisser une plaque — l'œil y lirait une
# texture qui bouge, pas des particules. Ici les vecteurs divergent, donc les
# points se croisent.
#
# Beaucoup de points très fins plutôt que quelques gros : dans un faisceau
# réel, la poussière est une nuée sous le seuil de netteté. Des points d'1,5 px
# se lisaient comme des pastilles.
#
# Le tirage est pseudo-aléatoire à graine fixe : il faut de l'irrégularité —
# une grille se verrait — mais stable d'un rendu à l'autre, sinon la nuée
# changerait de place à chaque rerun.
_DUST_COUNT = 8
_DUST_SEED = 20_250_104


def _dust_field():
    """(image, tailles, répétitions, départs, arrivées) des couches de points.

    `x` est tiré avec un exposant : les points se concentrent près du bord
    droit, où le faisceau est dense, et se raréfient en s'en éloignant.
    L'opacité suit la même décroissance, si bien qu'ils s'éteignent avec la
    lumière au lieu de flotter dans le noir.
    """
    rng = random.Random(_DUST_SEED)
    images, sizes, repeats, starts, ends = [], [], [], [], []
    for _ in range(_DUST_COUNT):
        far = rng.random() ** 1.7          # 0 au bord droit, 1 au plus loin
        x, y = 97.0 - far * 78.0, rng.uniform(4.0, 96.0)
        radius = round(rng.uniform(0.45, 0.95), 2)
        alpha = round((0.09 + 0.17 * (1.0 - far)) * rng.uniform(0.75, 1.15), 3)
        images.append(
            f"radial-gradient(circle, rgba(255,255,255,{alpha}) 0 {radius}px,"
            f" transparent {radius}px)"
        )
        sizes.append("10px 10px")
        repeats.append("no-repeat")
        starts.append(f"{x:.1f}% {y:.1f}%")
        ends.append(
            f"{min(max(x + rng.uniform(-26.0, 12.0), 0.0), 100.0):.1f}%"
            f" {min(max(y + rng.uniform(-75.0, 75.0), 0.0), 100.0):.1f}%"
        )
    return (",".join(images), ",".join(sizes), ",".join(repeats),
            ",".join(starts), ",".join(ends))


_DUST_IMAGE, _DUST_SIZE, _DUST_REPEAT, _DUST_START, _DUST_END = _dust_field()

# Ce que le cerclage projette : de nuit un faisceau qui décroît, de jour la
# peinture de la carrosserie, à plat.
# Pleins phares : au survol de la ligne déjà active, le faisceau gagne en
# intensité et porte plus loin vers la gauche. Le passage est net et non
# adouci — des phares basculent d'un coup, ils ne fondent pas.
# Flaque du clignotant : courte et resserrée, 28 % de la largeur contre 40 à
# 82 % pour le phare. Un clignotant éclaire ses abords, il n'ouvre pas la
# route — c'est ce rapport de portée qui empêche de les confondre.
_BLINK_POOL = ("linear-gradient(to left, "
               "color-mix(in srgb, var(--nyc-accent) 38%, transparent) 0%, "
               "color-mix(in srgb, var(--nyc-accent) 15%, transparent) 14%, "
               "transparent 28%)")

_LIT_HIGH = {
    "dark": ("linear-gradient(to left, "
             "color-mix(in srgb, var(--nyc-accent) 42%, transparent) 0%, "
             "color-mix(in srgb, var(--nyc-accent) 20%, transparent) 48%, "
             "color-mix(in srgb, var(--nyc-accent) 5%, transparent) 82%, "
             "transparent 100%)"),
}

_LIT = {
    "dark": ("linear-gradient(to left, "
             "color-mix(in srgb, var(--nyc-accent) 24%, transparent) 0%, "
             "color-mix(in srgb, var(--nyc-accent) 9%, transparent) 40%, "
             "transparent 100%)"),
    "light": ("linear-gradient(to left, "
              "color-mix(in srgb, var(--nyc-accent) 16%, transparent) 0 100%)"),
}

_GRID_LINE = {"dark": "rgba(255,255,255,0.07)", "light": "rgba(0,0,0,0.08)"}

# Alias historiques — DUSK_ORANGE et NIGHT_PURPLE retombent volontairement sur
# le même neutre.
_ALIASES = {
    "TAXI_GOLD": "ACCENT",
    "DOLLAR_GREEN": "BETTER",
    "DUSK_ORANGE": "NEUTRAL",
    "NIGHT_PURPLE": "NEUTRAL",
    "DUSK_RED": "WORSE",
    "GOLD_BORDER": "RULE",
}


@functools.lru_cache(maxsize=None)
def _load(m: str) -> dict:
    """Palette du mode `m`, lue dans config.toml.

    Les deux tables sont complètes : pas d'héritage à gérer depuis [theme],
    qui ne porte que des réglages de forme (rayons, bordures).
    """
    theme = tomllib.loads(_CONFIG.read_text(encoding="utf-8"))["theme"][m]
    p = {name: theme[opt] for name, opt in _MAP.items()}
    p["GRID_LINE"] = _GRID_LINE[m]
    return p


def _plate(pal: dict) -> str:
    """Teinte de la plaque de direction : l'accent posé sur le fond de barre.

    Opaque et non translucide — elle doit masquer le filet intérieur sur ses
    trois premiers pixels, ce qu'une couleur transparente ne ferait pas.
    """
    return f"color-mix(in srgb, {pal['ACCENT']} 17%, {pal['BG_PANEL']})"


def _rgb(hex_color: str) -> str:
    """#101014 -> "rgb(16, 16, 20)" — le format rendu par getComputedStyle."""
    h = hex_color.lstrip("#")
    return "rgb({}, {}, {})".format(*(int(h[i:i + 2], 16) for i in (0, 2, 4)))


def mode() -> str:
    """"light" ou "dark". Retombe sur "dark" hors runtime Streamlit."""
    try:
        return "light" if st.context.theme.type == "light" else "dark"
    except Exception:
        return "dark"


def palette() -> dict:
    return _load(mode())


def _band_ink() -> str:
    """Gris du bandeau : titre et icônes du commutateur, même valeur.

    Le gris secondaire tel quel passe pour noir sous un titre en capitales —
    une lettre pleine rend sa couleur, un trait de 2,2 px la dilue. Ramené de
    15 % vers le fond, le texte retrouve le poids visuel des icônes sans que
    les deux cessent d'être la même couleur.
    """
    p = palette()
    return f"color-mix(in srgb, {p['TEXT_DIM']} 85%, {p['BG_PAGE']})"


def fmt_int(v) -> str:
    """Entier avec espaces insécables — séparateur français, pas la virgule US."""
    return f"{v:,.0f}".replace(",", " ")


def _html(s: str) -> str:
    """Dédente le HTML multi-lignes avant st.markdown.

    Indispensable : même avec unsafe_allow_html, Markdown transforme toute
    ligne indentée d'au moins 4 espaces en bloc de code.
    """
    return re.sub(r"^[ \t]+", "", s, flags=re.MULTILINE)


def inject_css():
    """Ce que config.toml ne sait pas faire : typographie et sous-navigation.

    Fonds, bordures, rayons, pastille du radio et boutons sont désormais posés
    nativement par les tables [theme.light] / [theme.dark] — les régler ici en
    doublon les ferait diverger au premier changement de palette.
    """
    p = palette()
    st.markdown(
        f"""
        <style>
        /* L'accent en variable : les matières de la barre latérale sont des
           constantes de module, écrites avant que la palette soit connue. */
        .stApp {{ --nyc-accent: {p['ACCENT']}; }}

        /* Bandeau : fond plein et non plus transparent — le contenu défile
           dessous, et le groupe de titre posé sur du vide serait traversé
           par les graphiques. Le bouton de thème l'était déjà.

           Le bandeau ne couvre que la zone de contenu : son bord gauche est
           celui de la barre latérale. */
        [data-testid="stHeader"] {{
            background: {p['BG_PAGE']};
            border-bottom: 1px solid {p['RULE']};
        }}

        h1 {{ font-size: 1.6rem !important; font-weight: 600 !important;
              letter-spacing: -0.01em; color: {p['TEXT_LIGHT']} !important; }}
        h2 {{ font-size: 0.78rem !important; font-weight: 600 !important;
              letter-spacing: 0.1em; text-transform: uppercase;
              color: {p['TEXT_DIM']} !important; margin: 0 0 4px !important; }}
        h3 {{ font-size: 0.95rem !important; font-weight: 600 !important;
              color: {p['TEXT_LIGHT']} !important; }}

        [data-testid="stCaptionContainer"], .stMarkdown p {{ color: {p['TEXT_DIM']}; }}

        /* Éclairage de la barre, la nuit seulement : une nappe ambrée qui
           descend du haut, et de l'ombre qui s'accumule en bas. La lumière
           d'un lampadaire tombe d'en haut et s'épuise — un fond uniforme
           donnerait une pièce éclairée au néon, pas une rue.

           Deux dégradés fixes et aucun JavaScript : rien ne suit le curseur,
           rien ne se recalcule. Le coût est celui d'un fond peint une fois.
           De jour, le ciel éclaire tout également : pas de nappe. */
        [data-testid="stSidebar"] {{
            {"" if mode() == "light" else
             "background-image:"
             f" radial-gradient(130% 48% at 46% -10%,"
             f" color-mix(in srgb, {p['ACCENT']} 11%, transparent) 0%,"
             f" color-mix(in srgb, {p['ACCENT']} 3%, transparent) 42%,"
             " transparent 68%),"
             " radial-gradient(100% 38% at 50% 106%,"
             " rgba(0, 0, 0, 0.42) 0%, transparent 72%);"}
        }}


        /* Navigation — une option est un label [data-testid="stRadioOption"]
           qui contient la pastille puis le texte. Les sélecteurs
           [data-baseweb="radio"] de la 1.50 ont disparu en 1.66.

           La pastille est masquée : une case cochée est un formulaire, pas une
           navigation. La ligne active devient une surface — palier de fond
           teinté de l'accent, texte à l'accent — ce qui se repère d'un coup
           d'œil là où un repère de 16 px dans une colonne de 300 se cherche.
           Le label reste la cible de clic, l'input reste focusable. */
        /* Le groupe et ses lignes se dimensionnent sur leur texte : on les
           étire, sinon la surface active s'arrête au mot et ne se lit plus
           comme une ligne de navigation. */
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] {{
            gap: 2px !important;
            width: 100%;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {{ width: 100%; }}
        /* La chaîne s'arrête sur stElementContainer, qui se dimensionne sur
           son contenu : sans lui la largeur de 100 % ne vaut que celle du
           plus long intitulé. */
        [data-testid="stSidebar"] .stElementContainer:has(.stRadio) {{ width: 100% !important; }}
        [data-testid="stSidebar"] .stRadio {{ width: 100%; }}
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{
            display: none !important;
        }}
        /* Le conteneur interne est un flex : text-align n'y mord pas, il faut
           pousser son contenu en bout de ligne. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > div {{
            justify-content: flex-end;
        }}
        /* Navigation alignée à droite, contre la colonne de contenu : le
           bord vif se place là où va le regard, et le filet d'accent pointe
           vers ce qu'il commande au lieu de lui tourner le dos.

           Deux signaux pour la ligne active, pas un : un fond teinté et ce
           filet. La couleur seule se perd en vision périphérique, ce qui est
           précisément là où vit une barre latérale. Le filet existe toujours,
           transparent au repos, pour que rien ne se décale quand il s'allume. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"] {{
            position: relative;
            padding: 7px 22px 7px 12px;
            text-align: right;
            border-radius: 5px 0 0 5px;
            transition: background-color 0.18s ease, border-color 0.18s ease,
                        box-shadow 0.18s ease;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioOption"] p {{
            font-size: 0.9rem;
            color: {p['TEXT_DIM']};
            transition: color 0.18s ease;
        }}
        /* Au survol, le faisceau effleure sans se poser : même géométrie,
           en neutre et au tiers de l'intensité. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p {{
            color: {p['TEXT_LIGHT']};
        }}
        /* Couleur du libellé actif. Elle avait disparu dans une réécriture de
           ce bloc : le texte restait gris alors que tout le reste signalait
           la ligne. `!important` parce que Streamlit émet pour ses propres
           libellés des règles plus spécifiques que celle-ci. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p {{
            color: {p['ACCENT'] if mode() == "dark" else _SIGN_INK} !important;
            font-weight: 600;
            {"" if mode() == "dark" else
             "text-transform: uppercase; letter-spacing: 0.07em; font-size: 0.84rem;"}
        }}
        /* La ligne active emprunte au phare du taxi : un cerclage chromé, et
           la lumière qu'il projette. Le chrome est un dégradé vertical — bord
           sombre, arête claire, retour sombre — parce que c'est ainsi qu'un
           métal poli rend la lumière du ciel ; un gris uniforme ferait du
           plastique.

           De nuit, ce cerclage projette un faisceau ambré qui décroît vers la
           gauche. De jour on n'éclaire rien : la ligne prend la peinture de la
           carrosserie, à plat et sans dégradé. Même objet, deux lectures
           selon l'heure. D'où l'alignement à droite dans les deux cas : le
           texte se tient contre le métal.

           Deux couches de `background-image` plutôt qu'une bordure : une
           bordure ne sait pas porter de dégradé. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {{
            background-image: {_LIT[mode()]};
            padding-right: 18px;
            overflow: hidden;
        }}
        /* Les poussières dérivent, sinon ce sont des taches. Lentement et
           sur quelques pixels : de la matière en suspension, pas une pluie.
           Elles vivent sur leur propre couche pour que le faisceau, lui, ne
           bouge pas — une seule déclaration de fond ferait glisser les deux.
           Le débord de 10 px donne de la course au mouvement, et le
           `overflow: hidden` de la ligne en masque les bords. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked)::before {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background-image: {_DUST_IMAGE if mode() == "dark" else "none"};
            background-size: {_DUST_SIZE};
            background-repeat: {_DUST_REPEAT};
            background-position: {_DUST_START};
            animation: nyc-dust 7s linear infinite alternate;
        }}
        /* Dérive linéaire et non adoucie : une poussière portée par l'air
           garde sa vitesse, elle ne ralentit pas aux extrémités. */
        @keyframes nyc-dust {{
            from {{ background-position: {_DUST_START}; }}
            to   {{ background-position: {_DUST_END}; }}
        }}

        /* Une dérive de poussière n'est jamais indispensable à la lecture. */
        @media (prefers-reduced-motion: reduce) {{
            [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked)::before {{
                animation: none;
            }}
        }}

        /* De jour il n'y a pas de phare à allumer : l'aplat et le filet sont
           portés par la plaque elle-même, juste en dessous. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {{
            {"" if mode() == "dark" else "background-image: none;"}
        }}

        /* Pleins phares : survoler la ligne active monte l'intensité. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked):hover {{
            {f"background-image: {_LIT_HIGH['dark']};" if mode() == "dark" else ""}
        }}

        /* Clignotant : survoler une destination où l'on ne va pas encore,
           c'est signaler son intention d'y aller. Un feu ambré au même
           emplacement que le cerclage, qui bat à 1,5 Hz — la cadence d'un
           vrai clignotant, entre 60 et 120 battements par minute.

           Les transitions occupent 6 % du cycle, soit 41 ms : assez court
           pour qu'une ampoule s'allume, assez long pour éviter la coupure
           franche qui ferait stroboscope. */
        /* La lumière que l'ampoule projette, sur sa propre couche pour battre
           avec elle : une couleur de fond ne s'anime pas en cadence. */
        [data-testid="stSidebar"]
            [data-testid="stRadioOption"]:not(:has(input:checked)):hover::before {{
            {"" if mode() == "light" else
             'content: ""; position: absolute;'
             ' left: 0; right: 0; top: 50%; height: 18px; margin-top: -9px;'
             ' pointer-events: none;'
             f' background-image: {_BLINK_POOL};'
             ' animation: nyc-blink 0.68s linear infinite;'}
        }}

        [data-testid="stSidebar"]
            [data-testid="stRadioOption"]:not(:has(input:checked)):hover::after {{
            content: {'""' if mode() == "dark" else "none"};
            position: absolute;
            top: 50%; right: 0;
            width: 5px; height: 18px;
            margin-top: -9px;
            border-radius: 2.5px;
            background: {p['ACCENT']};
            /* Une ampoule éclaire ce qui l'entoure. Deux halos concentriques :
               un proche et dense, un large et diffus — c'est la décroissance
               d'une source ponctuelle. La lueur est admise ici parce qu'elle
               a une source ; posée sur une carte, elle ne serait qu'un effet.
               Elle bat avec l'ampoule, puisque l'opacité porte les deux. */
            box-shadow: 0 0 5px 0 color-mix(in srgb, {p['ACCENT']} 85%, transparent),
                        0 0 16px 3px color-mix(in srgb, {p['ACCENT']} 45%, transparent);
            animation: nyc-blink 0.68s linear infinite;
        }}
        @keyframes nyc-blink {{
            0%, 42% {{ opacity: 1; }}
            48%, 100% {{ opacity: 0; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            [data-testid="stSidebar"]
                [data-testid="stRadioOption"]:not(:has(input:checked)):hover::after {{
                animation: none;
                opacity: 0.55;
            }}
        }}

        /* De jour, survoler une destination fait apparaître la même flèche,
           après le mot. Elle s'insère dans le flux et pousse donc le libellé
           vers la gauche — c'est voulu : le mot recule devant la flèche qui
           le désigne.

           Masque et non glyphe : le caractère d'une police est un trait fin,
           là où une flèche de signalisation a un fût épais et une tête
           triangulaire pleine. La même forme sert ensuite en blanc sur la
           plaque, puisque c'est la couleur de fond qui la teinte. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:not(:has(input:checked)):hover
            [data-testid="stMarkdownContainer"] p::after {{
            content: {'none' if mode() == "dark" else '""'};
            display: inline-block;
            width: 20px; height: 10px;
            margin-left: 10px;
            vertical-align: 0.08em;
            transform: scaleX(-1);
            background-color: {p['TEXT_LIGHT']};
            -webkit-mask: {_SIGN_ARROW_MASK};
            mask: {_SIGN_ARROW_MASK};
            animation: nyc-arrow 0.2s ease-out;
        }}

        /* La plaque garde sa flèche, en blanc, après le mot et pointée vers
           lui : sur un panneau elle indique la direction à prendre, pas le
           sens de lecture. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked)
            [data-testid="stMarkdownContainer"] p::after {{
            content: {'none' if mode() == "dark" else '""'};
            display: inline-block;
            width: 20px; height: 10px;
            margin-left: 10px;
            vertical-align: 0.08em;
            transform: scaleX(-1);
            background-color: {_SIGN_INK};
            -webkit-mask: {_SIGN_ARROW_MASK};
            mask: {_SIGN_ARROW_MASK};
        }}

        @keyframes nyc-arrow {{
            from {{ opacity: 0; transform: scaleX(-1) translateX(-7px); }}
            to   {{ opacity: 1; transform: scaleX(-1) translateX(0); }}
        }}

        /* Le cerclage est une pièce, pas un trait : une épaisseur, des bouts
           arrondis, et un retrait haut et bas qui le détache des bords de la
           ligne. En couche de fond il restait un rectangle collé au bord. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked)::after {{
            {"" if mode() == "light" else
             'content: ""; position: absolute;'
             ' top: 1px; bottom: 1px; right: 0; width: 3px; border-radius: 2px;'
             f' background: {_METAL["dark"]};'
             ' box-shadow: inset 0 0 0 0.5px rgba(255,255,255,0.08);'}
        }}

        /* Au survol, le mot prend la typographie du panneau — capitales et
           interlettrage — sans son fond. C'est l'aperçu de ce que donnera la
           sélection : le même langage à deux intensités, et le changement de
           chasse fournit le mouvement.

           L'essai concurrent, une peinture verte balayant depuis la droite,
           a été écarté : le vert passait sous le texte noir, qui devenait
           illisible sur la moitié peinte. Faire basculer l'encre en blanc à
           l'avancée de la peinture demanderait de dupliquer le texte en deux
           calques masqués. */
        [data-testid="stSidebar"]
            [data-testid="stRadioOption"]:not(:has(input:checked)):hover p {{
            {"" if mode() == "dark" else
             "text-transform: uppercase; letter-spacing: 0.07em;"
             " font-size: 0.84rem; font-weight: 600;"}
        }}

        /* La plaque ne fait pas toute la largeur : elle est à la taille de son
           texte, comme une vraie plaque de rue. C'est le conteneur interne
           qui la porte, et non la ligne entière, sinon l'aplat s'étirerait
           jusqu'au bord de la barre. */
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) > div {{
            {"" if mode() == "dark" else
             f'display: inline-flex; width: auto; padding: 7px 30px;'
             f' margin-right: -14px;'
             f' flex-direction: row; align-items: center; white-space: nowrap;'
             f' background-image: {_METAL_BAND}, {_METAL_BAND}, {_SIGN_FACE};'
             ' background-size: 100% 3px, 100% 3px, 100% 100%;'
             ' background-position: top left, bottom left, center;'
             ' background-repeat: no-repeat, no-repeat, no-repeat;'}
        }}

        /* Second groupe — À propos et Pipeline de données, à partir de la
           cinquième option. Séparés par un filet et un blanc, rendus d'un
           cran plus petits : ce sont des pages de contexte, pas la raison
           d'être du site. nth-child doit rester accordé à N_PRIMARY. */
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(5) {{
            margin-top: 22px;
            padding-top: 16px;
            /* Marquage au sol : traits longs, intervalles longs, dans la
               teinte des filets du reste de l'interface. Un dégradé répété
               plutôt qu'un `border-top: dashed`, dont le navigateur choisit
               seul la longueur des tirets. */
            background-image: repeating-linear-gradient(to right,
                {p['RULE']} 0 20px, transparent 20px 34px);
            background-size: 100% 2px;
            background-position: left top;
            background-repeat: no-repeat;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(n+5)
            [data-testid="stRadioOption"] {{
            padding-top: 5px; padding-bottom: 5px;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(n+5)
            [data-testid="stRadioOption"] p {{
            /* Un cran plus petit encore que le reste du second groupe : la
               barre est à sa largeur minimale (200 px) et « Pipeline de
               données » doit y tenir sur une ligne sans la déborder. */
            font-size: 0.72rem;
            white-space: nowrap;
        }}


        /* Rappel de périmètre : une ligne sous le titre, assez discrète pour
           ne pas concurrencer les indicateurs qui suivent. Remonte contre le
           titre — Streamlit laisse sinon un blanc qui la détache du bloc. */
        .scope-strip {{
            margin: -10px 0 18px;
            font-size: 0.74rem;
            color: {p['TEXT_DIM']};
            font-variant-numeric: tabular-nums;
        }}

        /* Sélecteur de section : la même discrétion que la navigation, des
           libellés un cran plus petits que le corps de texte. */
        [data-testid="stSegmentedControl"] button p {{ font-size: 0.82rem; }}

        /* Contrôle de thème remonté dans la bannière native. Même technique
           que qlt-eau-FR-24 : position fixe sur le conteneur à clé, et
           z-index au-dessus de celui du header (999990). La bannière fait
           60 px de haut, d'où le centrage vertical.

           152 px de décalage et non 56 : l'indicateur d'exécution que
           Streamlit insère pendant chaque rerun occupe 48 à 141 px depuis le
           bord droit (mesuré), et recouvrait le contrôle à chaque bascule.

           Le bouton Deploy est masqué : il n'apparaît qu'en local, et sans
           lui la bannière a la même géométrie partout. */
        [data-testid="stAppDeployButton"] {{ display: none !important; }}

        /* Le menu ☰ sort de la vue : la navigation est dans la barre, le
           thème dans son commutateur, et ses entrées restantes — Rerun,
           Clear cache, Print — sont des outils de développement.

           Sorti du champ et non `display: none` : le commutateur de thème
           clique l'entrée de thème de ce menu pour basculer sans recharger
           la page. Masqué pour de bon, il n'y aurait plus rien à cliquer et
           chaque bascule repasserait par un rechargement. */
        [data-testid="stMainMenu"] {{
            position: absolute !important;
            left: -9999px !important;
            top: 0 !important;
        }}
        /* Le panneau du menu, lui, ne suit pas son bouton hors du champ : il
           est rabattu dans la fenêtre et se dessinait en haut à gauche le
           temps que le commutateur de thème le pilote. Invisible et
           intraversable, mais toujours rendu — `display: none` le retirerait
           du flux et le clic programmatique n'aurait plus de cible. */
        [data-testid="stMainMenuPopover"] {{
            opacity: 0 !important;
            pointer-events: none !important;
        }}

        /* Dans le commutateur de thème du menu ☰, Streamlit peint le focus et
           la sélection avec le même aplat, à deux opacités près (0.15 et
           0.25) : deux entrées paraissent actives et l'on ne sait plus laquelle
           est choisie. Le focus passe donc en contour, la sélection garde
           l'aplat. Le cas se présente après un clic de notre bouton, qui
           ouvre ce menu par programme et laisse le navigateur en modalité
           clavier, mais la confusion existait déjà au clavier seul. */
        [data-testid="stThemeSwitcher"] button[aria-checked="false"]:focus-visible {{
            background-color: transparent !important;
            outline: 2px solid {p['ACCENT']};
            outline-offset: -2px;
        }}
        /* Groupe de titre : logo, nom, commutateur, dans cet ordre et sur
           une seule ligne, calés contre la barre latérale.

           Un seul élément porte les trois, donc un seul gap les sépare et
           rien n'est à recalculer si le nom change. Le logo et le nom sont
           des pseudo-éléments, remis dans l'ordre par `order` puisque
           ::after se place sinon après l'iframe du commutateur.

           La position est fixe, mais son abscisse vient de --nyc-band-x, que
           le composant tient à jour : la barre latérale est redimensionnable
           à la souris, et une valeur en dur se décrocherait du bord dès le
           premier glissement. 340px en secours = 320 de barre + 20 de marge,
           l'état au chargement. */
        .st-key-band {{
            position: fixed !important; top: 0 !important;
            left: var(--nyc-band-x, 340px) !important; right: auto !important;
            height: 60px !important; width: auto !important;
            display: flex !important; flex-direction: row !important;
            align-items: center !important; gap: 0.625rem !important;
            z-index: 1000001 !important;
        }}
        .st-key-band::before {{
            order: 1;
            content: "";
            width: 1.75rem; height: 1.75rem;
            background: {_LOGO_URI} center / contain no-repeat;
        }}
        .st-key-band::after {{
            order: 2;
            content: "{nyc_lang.t('band.title')}";
            margin-right: 0.625rem;
            font-size: 0.8125rem;
            /* 600 et non 650 : en capitales, un gras appuyé fait lire le gris
               plus sombre qu'il n'est. */
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            white-space: nowrap;
            /* Même gris que les icônes du commutateur, qui le prennent en
               currentColor : un seul ton pour tout le bandeau. */
            color: {_band_ink()};
        }}
        /* Les deux commandes gardent l'ordre du code — commutateur de thème,
           puis langues — puisqu'elles partagent la même valeur d'`order` et
           viennent après les pseudo-éléments. */
        .st-key-band > * {{ order: 3; }}
        /* Streamlit donne au conteneur du composant une base de 26 px — la
           hauteur passée à components.html. L'iframe, elle, fait 72 px de
           large et débordait sur le sélecteur de langue. Les deux conteneurs
           sont donc dimensionnés ici : largeur fixe pour l'iframe, contenu
           pour le reste. */
        .st-key-band [data-testid="stElementContainer"] {{
            flex: 0 0 auto !important;
            width: auto !important;
        }}
        .st-key-band [data-testid="stElementContainer"]:has(iframe) {{
            flex: 0 0 72px !important;
            width: 72px !important;
        }}
        .st-key-band iframe {{ display: block; width: 72px !important; }}

        /* Sélecteur de langue : trois intitulés, pas de pastilles. Le même
           gris que le reste du bandeau, l'actif en plein, les autres en
           retrait — une commande qui ne se dispute pas la lecture avec le
           titre à côté. */
        .st-key-band [data-testid="stRadioGroup"] {{
            flex-direction: row !important;
            gap: 0.5rem !important;
            align-items: center;
        }}
        .st-key-band [data-testid="stRadioOption"] > div > div:first-child {{
            display: none !important;
        }}
        .st-key-band [data-testid="stRadioOption"] p {{
            font-size: 0.6875rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {_band_ink()} !important;
            opacity: 0.55;
        }}
        .st-key-band [data-testid="stRadioOption"]:hover p {{ opacity: 0.8; }}
        .st-key-band [data-testid="stRadioOption"]:has(input:checked) p {{
            opacity: 1;
        }}

        /* Déclencheur de rerun du bouton de thème : présent dans le DOM pour
           rester cliquable depuis le JavaScript, mais hors de la page. */
        .st-key-theme_rerun {{
            position: absolute; left: -9999px; top: 0;
            width: 1px; height: 1px; overflow: hidden;
        }}
        .stButton>button {{ font-size: 0.8rem; font-weight: 500; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Le JavaScript est tenu hors f-string : ses accolades devraient toutes
# être doublées, ce qui le rendrait illisible et fragile à la retouche.
# Icônes du contrôle de thème. `currentColor` les fait suivre l'état de leur
# cellule (accentuée ou non), et `aria-hidden` évite qu'un lecteur d'écran les
# annonce en plus du nom accessible du bouton.
# Icônes du thème : deux pastilles de même diamètre, pleine pour le clair et
# creuse pour le sombre — le disque plein se lit « éclairé », le cercle vide
# « éteint ». Les rayons sont accordés (9 contre 7.9 + 1.1 de trait) pour que
# les deux fassent exactement la même tache. Pas de demi-disque : Streamlit
# s'en sert pour « System » dans son menu ☰, et un interrupteur à deux
# positions demande deux pôles opposés. `currentColor` les fait suivre le
# thème.
_ICON_LIGHT = (
    '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9" fill="currentColor"/></svg>'
)
_ICON_DARK = (
    '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">'
    '<circle cx="12" cy="12" r="7.9" fill="none" stroke="currentColor" stroke-width="2.2"/>'
    '</svg>'
)


_TOGGLE_JS = r"""
var KEY2 = 'stActiveTheme-/-v2';
var KEY1 = 'stActiveTheme-/-v1';
var sw = document.getElementById('sw');

function doc() { return window.parent.document; }
function ls() { return window.parent.localStorage; }

// Ce que Streamlit a retenu : 'System', 'Light', 'Dark', ou rien.
function stored() {
  try { var raw = ls().getItem(KEY2); return raw ? JSON.parse(raw) : null; }
  catch (e) { return null; }
}

function system() {
  try {
    return window.parent.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'Dark' : 'Light';
  } catch (e) { return 'Dark'; }
}

// Thème en vigueur : 'System' et l'absence de clé se résolvent tous deux sur
// la préférence du navigateur.
function effective() {
  var v = stored();
  return (v === 'Light' || v === 'Dark') ? v : system();
}

function write(v) {
  ls().setItem(KEY2, JSON.stringify(v));
  ls().setItem(KEY1, JSON.stringify({name: v}));
}

// Fond réellement peint par le front : seul juge fiable du thème affiché.
function painted() {
  try {
    return window.parent.getComputedStyle(doc().querySelector('.stApp')).backgroundColor;
  } catch (e) { return null; }
}

// Les deux icônes encadrent le rail : le contrôle dit où l'on est par la
// position de la pastille, et ce que l'on peut faire par l'icône opposée.
// `role="switch"` plutôt qu'une case à cocher — c'est bien un état binaire
// appliqué sur-le-champ, pas un réglage à valider.
function render(v) {
  var dark = v === 'Dark';
  sw.setAttribute('aria-checked', String(dark));
  sw.setAttribute('aria-label', dark ? 'Thème sombre' : 'Thème clair');
}

// Clic « à la souris ». element.click() produit un évènement de detail 0, que
// le navigateur range du côté clavier et qui laisse un :focus-visible derrière
// lui. Un MouseEvent de detail 1 est lu comme un vrai clic.
function tap(el) {
  el.dispatchEvent(new window.parent.MouseEvent('click', {
    bubbles: true, cancelable: true, view: window.parent, detail: 1,
  }));
}

// Relance le script Python sans recharger la page, via le bouton Streamlit
// posé hors écran par band().
function rerun() {
  var b = doc().querySelector('.st-key-theme_rerun button');
  if (!b) { return false; }
  b.click();
  return true;
}

// Repli commun à tous les échecs : écrire la clé et recharger, ce qui marche
// sans rien connaître du DOM de Streamlit.
function fallback(target) {
  try { write(target); } catch (e) {}
  window.parent.location.reload();
}

// Bascule par le commutateur natif du menu ☰ (data-testid stables depuis la
// 1.66). Lui seul repeint les widgets Streamlit sans rechargement. Le menu
// doit être ouvert pour que ses entrées soient montées, d'où l'attente.
function switchNative(target, done) {
  var menu = doc().querySelector('[data-testid="stMainMenuButton"]');
  if (!menu) { return done(false); }
  tap(menu);

  var tries = 0;
  (function wait() {
    try {
      var item = doc().querySelector('[data-testid="stMainMenuItem-theme-' + target + '"]');
      if (item) {
        tap(item);
        // Le menu ne se referme pas tout seul après un clic sur le thème.
        if (doc().querySelector('[data-testid="stThemeSwitcher"]')) { tap(menu); }
        var active = doc().activeElement;
        if (active && active.blur) { active.blur(); }
        return done(rerun());
      }
      if (++tries > 60) { return done(false); }
      setTimeout(wait, 16);
    } catch (e) { done(false); }
  })();
}

// Le front peut changer de thème sans que le script Python soit rejoué :
// c'est le cas du commutateur natif du menu ☰, et du premier affichage quand
// le serveur peint avant que la préférence système soit résolue. Les widgets
// natifs basculent alors seuls, tandis que notre HTML et nos figures Plotly
// restent dans l'ancien thème — une page à moitié repeinte.
//
// On compare donc en continu le fond réellement appliqué au fond que le
// serveur a utilisé pour ce rendu (SERVER_BG). Dès qu'ils divergent, on
// relance le script.
//
// `syncing` n'autorise qu'une relance par montage du composant. Un rerun
// remonte l'iframe et remet donc le drapeau à faux : les bascules suivantes
// sont servies normalement. Mais si la relance ne résout pas l'écart — thème
// « Custom Theme » du menu, dont le fond ne sera jamais l'un des nôtres — le
// drapeau reste levé et rien ne boucle.
var syncing = false;

function syncIfDrifted() {
  if (syncing) { return; }
  var now = painted();
  if (!now || now === SERVER_BG) { return; }
  syncing = true;
  if (!rerun()) { window.parent.location.reload(); }
}

var current = effective();

sw.addEventListener('click', function () {
  var target = effective() === 'Dark' ? 'Light' : 'Dark';
  try {
    switchNative(target, function (ok) { if (!ok) { fallback(target); } });
  } catch (e) { fallback(target); }
});

render(current);

try {
  // Le thème se lit dans la classe que Streamlit pose sur .stApp : l'observer
  // plutôt que sonder en boucle. Le court délai laisse les styles s'appliquer
  // avant qu'on lise le fond.
  var app = doc().querySelector('.stApp');
  if (app) {
    new window.parent.MutationObserver(function () {
      window.parent.setTimeout(syncIfDrifted, 60);
    }).observe(app, {attributes: true, attributeFilter: ['class', 'style']});
  }
  syncIfDrifted();
} catch (e) {}

// Abscisse du groupe de titre. Le groupe est en position fixe — seul moyen de
// le sortir du flux de la page pour le poser dans le bandeau — mais son point
// d'ancrage bouge : la barre latérale se replie, et le visiteur peut la tirer
// par son bord. Le bandeau, lui, commence toujours au bord de la barre : son
// abscisse est donc la mesure cherchée, et un ResizeObserver sur lui voit
// aussi bien le glissement que le repli.
//
// Barre ouverte : 20 px de marge, celle que le logo prenait dans la barre.
// Barre repliée : le bandeau part de 0 et Streamlit y remet son bouton de
// dépli, large de 28 px contre le bord — le groupe recule à 64 px.
function placeBand() {
  try {
    var d = doc();
    var h = d.querySelector('[data-testid="stHeader"]');
    if (!h) { return; }
    var x = Math.round(h.getBoundingClientRect().left);
    d.documentElement.style.setProperty('--nyc-band-x', (x + (x > 0 ? 20 : 64)) + 'px');
  } catch (e) {}
}

try {
  placeBand();
  var band = doc().querySelector('[data-testid="stHeader"]');
  if (band && window.parent.ResizeObserver) {
    new window.parent.ResizeObserver(placeBand).observe(band);
  }
  window.parent.addEventListener('resize', placeBand);
} catch (e) {}
"""


def band():
    """Le groupe de la barre de titre : logo, nom, thème, langue.

    Le logo et le nom sont des pseudo-éléments posés par `inject_css()` sur le
    conteneur rendu ici ; seules les deux commandes sont des widgets.

    La bascule de thème, elle, est clair / sombre, et par défaut le thème du
    navigateur.

    Double le sélecteur natif du menu ☰, qui propose depuis la 1.66 les trois
    états System / Light / Dark. Le menu reste la référence : ce bouton ne fait
    que cliquer la même entrée pour lui, les deux commandes restent donc
    d'accord. Il existe parce que le menu est à trois clics et hors du champ de
    lecture.

    Pourquoi ne pas l'écrire en Python : `st.context.theme` est en lecture
    seule, et `_config.set_option` agirait côté serveur, donc pour tous les
    visiteurs à la fois — inacceptable sur un site public.

    Le choix est stocké par le front dans le localStorage du navigateur, sous
    `stActiveTheme-/-v2`, avec trois valeurs : "System", "Light" ou "Dark".
    « System » et l'absence de clé se résolvent tous deux sur
    `prefers-color-scheme` : c'est ce qui fait du thème navigateur le défaut.
    Tant que le visiteur n'a pas cliqué, la clé reste sur "System" et continue
    donc de suivre son navigateur.

    Mécanique, en deux temps. Le composant tourne dans un iframe `srcdoc` de
    même origine, il atteint donc le DOM du parent. Il ouvre le menu, clique
    l'entrée de thème voulue — le front repeint alors les widgets natifs tout
    seul — puis referme le menu. Reste notre moitié à nous : un changement de
    thème ne déclenche aucun rerun (vérifié en 1.66), si bien que le HTML des
    cartes et les couleurs Plotly, calculés côté serveur, resteraient dans
    l'ancien thème. D'où le second temps : cliquer le bouton Streamlit posé
    hors écran, qui relance le script sans recharger la page.

    Les `data-testid` visés — `stMainMenuButton`, `stThemeSwitcher`,
    `stMainMenuItem-theme-*` — sont du balisage interne. S'ils disparaissent,
    `switchNative` rend la main sur `false` et l'on retombe sur `fallback` :
    écrire la clé puis recharger la page, ce qui marche sans rien connaître du
    DOM de Streamlit. La navigation survit à ce rechargement grâce aux
    paramètres d'URL.

    Un changement de thème peut aussi venir d'ailleurs : du menu natif, ou du
    premier affichage quand le serveur peint avant que la préférence système
    soit résolue. Sans rerun, la page reste à moitié repeinte. Le composant
    surveille donc le fond réellement appliqué sur `.stApp` et relance dès
    qu'il s'écarte de celui que le serveur a utilisé — le juge n'est pas
    `st.context.theme.type`, qui est justement ce qui peut être faux.
    """
    import streamlit.components.v1 as components

    p = palette()
    css = f"""
        html, body {{ margin:0; padding:0; background:transparent; height:100%; }}
        /* Rail coulissant encadré des deux icônes, repris de qlt-eau-FR-24 :
           14 px d'icône, 6 px de marge, rail de 32x16. Là-bas c'est un
           st.toggle natif ; ici il faut un interrupteur à nous, parce que le
           clic doit passer par le commutateur de thème de Streamlit et non
           par un rerun Python. */
        #g {{ display: flex; align-items: center; height: 100%;
              color: {_band_ink()}; }}
        .ic {{ display: block; line-height: 0; }}
        .ic.s {{ margin-right: 6px; }}
        .ic.m {{ margin-left: 6px; }}
        #sw {{
          all: unset;
          box-sizing: border-box;
          display: inline-flex; align-items: center;
          width: 32px; height: 16px; padding: 2px;
          border-radius: 999px; background: {p['BAR_MUTED']};
          cursor: pointer;
        }}
        #sw[aria-checked="true"] {{ background: {p['NEUTRAL']}; }}
        #sw:focus-visible {{ outline: 2px solid {p['ACCENT']}; outline-offset: 2px; }}
        .knob {{
          width: 12px; height: 12px; border-radius: 50%;
          background: {p['BG_PAGE']};
          transform: translateX(0);
          transition: transform 0.15s ease;
        }}
        #sw[aria-checked="true"] .knob {{ transform: translateX(16px); }}
        /* Sans ceci, le survol vise le SVG ou la pastille, qui n'héritent pas
           du `title` de l'interrupteur : l'infobulle ne sortirait jamais. */
        #sw svg, .knob, .ic {{ pointer-events: none; }}
    """
    # Cible du rerun déclenché depuis le JavaScript. Le conteneur porte une
    # clé pour obtenir la classe st-key-theme_rerun, que le CSS sort du champ
    # de vision sans la retirer du DOM — un display:none la rendrait
    # inatteignable au clic.
    with st.container(key="theme_rerun"):
        st.button("Recharger le thème", key="_theme_rerun")

    # Fond que ce rendu-ci utilise : la référence contre laquelle le composant
    # détecte qu'un changement de thème n'a pas été suivi d'un rerun.
    js = f'var SERVER_BG = "{_rgb(palette()["BG_PAGE"])}";' + _TOGGLE_JS
    # Conteneur à clé : produit la classe st-key-band, sur laquelle le CSS
    # accroche la position fixe qui pose le groupe dans la bannière, ainsi que
    # le logo et le titre en pseudo-éléments.
    with st.container(key="band"):
        components.html(
            (
            f'<style>{css}</style>'
            f'<div id="g">'
            f'<span class="ic s">{_ICON_LIGHT}</span>'
            f'<button id="sw" role="switch"><span class="knob"></span></button>'
            f'<span class="ic m">{_ICON_DARK}</span>'
            f'</div><script>{js}</script>'
        ),
            height=26,
        )
        _lang_picker()


def _lang_picker():
    """Trois intitulés de langue, dont un seul est actif.

    Un `st.radio` et non trois boutons : l'état est unique et porté par le
    widget, là où trois boutons demanderaient d'arbitrer lequel a été cliqué
    en dernier. Les pastilles sont masquées en CSS — une case cochée est un
    formulaire, pas une commande de bandeau.

    L'écriture dans `?lang=` suffirait à relancer le script, mais le `rerun`
    explicite évite de dépendre de ce détail d'implémentation. Pas de boucle :
    au tour suivant, `current()` renvoie la langue qu'on vient d'écrire et la
    condition est fausse.
    """
    choice = st.radio(
        "Langue", nyc_lang.LANGS,
        index=nyc_lang.LANGS.index(nyc_lang.current()),
        format_func=str.upper, horizontal=True,
        label_visibility="collapsed", key="band_lang",
    )
    if choice != nyc_lang.current():
        st.query_params["lang"] = choice
        st.rerun()


def sparkline(values, color=None, height=22):
    """Courbe miniature en SVG inline — pas de Plotly pour une telle vignette.

    `preserveAspectRatio="none"` étire le tracé sur toute la largeur de la
    carte ; `vector-effect` garde malgré tout un trait d'épaisseur constante.
    """
    color = color or palette()["ACCENT"]
    values = [float(v) for v in values]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    w = 120.0
    pts = " ".join(
        f"{i / (len(values) - 1) * w:.1f},{height - (v - lo) / span * (height - 3) - 1.5:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 {w:.0f} {height}" preserveAspectRatio="none" '
        f'style="width:100%; height:{height}px; display:block; margin:6px 0 2px;">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2" '
        f'vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>'
    )


def delta_badge(pct):
    """Variation en pourcentage, flèche portant le sens.

    Teinte neutre volontairement : sur ces indicateurs, « en hausse » ne veut
    pas dire « mieux » (une distance moyenne qui monte n'est pas une bonne
    nouvelle en soi). Colorer en vert/rouge y plaquerait un jugement faux.
    """
    if pct is None:
        return ""
    arrow = "▲" if pct >= 0 else "▼"
    return (
        f'<span style="font-size:0.8rem; color:{palette()["TEXT_DIM"]}; white-space:nowrap; '
        f'font-variant-numeric:tabular-nums;">{arrow} {abs(pct):.1f}&nbsp;%</span>'
    )


def card(label, value, detail="", color=None, icon="", delta=None, spark=None):
    """Indicateur : libellé, valeur (+ variation), courbe, filet, précision.

    `icon` n'est plus rendu — le paramètre reste pour ne pas casser les appels.
    """
    p = palette()
    color = color or p["TEXT_LIGHT"]
    spark_html = sparkline(spark) if spark is not None else ""
    st.markdown(
        _html(f"""<div style="padding:0 0 18px;">
              <div style="font-size:0.66rem; font-weight:600; letter-spacing:0.09em;
                          text-transform:uppercase; color:{p['TEXT_DIM']};">{label}</div>
              <div style="font-size:1.7rem; font-weight:600; color:{color};
                          line-height:1.25; margin-top:2px;
                          font-variant-numeric:tabular-nums;">{value}</div>
              <div style="min-height:1.1rem;">{delta_badge(delta)}</div>
              {spark_html}
              <div style="border-top:1px solid {p['RULE']}; margin:8px 0 6px;"></div>
              <div style="font-size:0.72rem; color:{p['TEXT_DIM']};">{detail}</div>
            </div>"""),
        unsafe_allow_html=True,
    )


# La police du titre passe par un template et non par update_layout : les
# graphiques appellent style_fig() PUIS update_layout(title="…"), ce qui
# remplace tout l'objet `title` et effacerait la police. Un template, lui,
# n'est pas écrasé. Le mettre dans le layout poserait un second problème :
# un `title` sans `text` est rendu littéralement "undefined" par Plotly.js.
_TEMPLATE_CACHE = {}


def _plot_template():
    m = mode()
    if m not in _TEMPLATE_CACHE:
        base = "plotly_white" if m == "light" else "plotly_dark"
        tmpl = go.layout.Template(pio.templates[base])
        tmpl.layout.title.font = dict(size=12, color=palette()["TEXT_DIM"])
        _TEMPLATE_CACHE[m] = tmpl
    return _TEMPLATE_CACHE[m]


def style_fig(fig, **overrides):
    """Applique le thème aux figures Plotly (à appeler avant les update_layout
    spécifiques à chaque graphique — le merge est récursif)."""
    p = palette()
    fig.update_layout(
        template=_plot_template(),
        paper_bgcolor=p["BG_PAGE"],
        plot_bgcolor=p["BG_PAGE"],
        font=dict(color=p["TEXT_LIGHT"], size=11),
        xaxis=dict(gridcolor=p["GRID_LINE"], zerolinecolor=p["GRID_LINE"],
                   linecolor=p["RULE"]),
        yaxis=dict(gridcolor=p["GRID_LINE"], zerolinecolor=p["GRID_LINE"],
                   linecolor=p["RULE"]),
        **overrides,
    )
    return fig


def __getattr__(name):
    """Résout les noms de couleurs sur le thème actif (PEP 562).

    Permet à `nyc_theme.ACCENT` de rester écrit tel quel chez les appelants
    tout en renvoyant la valeur du thème courant. N'est consulté que si la
    recherche normale échoue : aucune des couleurs ne doit donc exister comme
    variable de module.
    """
    key = _ALIASES.get(name, name)
    p = palette()
    if key in p:
        return p[key]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
