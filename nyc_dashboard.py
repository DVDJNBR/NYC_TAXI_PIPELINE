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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nyc_content
import nyc_theme

PAGE_CONFIG = dict(
    page_title="NYC Yellow Taxi",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_ABOUT    = "À propos"
PAGE_PIPELINE = "Pipeline de données"


# Le dashboard est découpé en questions plutôt qu'en catégories : court,
# mémorisable, et traduisible sans perte (How much / When / Where / How).
SEC_HOW_MUCH = "Combien"
SEC_WHEN     = "Quand"
SEC_WHERE    = "Où"
SEC_HOW      = "Comment"
SECTIONS = [SEC_HOW_MUCH, SEC_WHEN, SEC_WHERE, SEC_HOW]

# Une seule liste pour la barre : les sections du dashboard en tête, les pages
# de contexte en dessous. Le CSS distingue les deux groupes en visant les
# options au-delà de la quatrième — d'où N_PRIMARY, qui doit rester accordé.
VIEWS = SECTIONS + [PAGE_ABOUT, PAGE_PIPELINE]
N_PRIMARY = len(SECTIONS)
# "Quand ?" est la section la plus riche : c'est elle qui ouvre.
DEFAULT_SECTION = SEC_WHEN

# ---------------------------------------------------------------------------
# Lookup officiel TLC : location_id -> nom de quartier
# ---------------------------------------------------------------------------
ZONE_LOOKUP = {
    1: "Newark Airport", 2: "Jamaica Bay", 3: "Allerton/Pelham Gardens",
    4: "Alphabet City", 5: "Arden Heights", 6: "Arrochar/Fort Wadsworth",
    7: "Astoria", 8: "Astoria Park", 9: "Auburndale", 10: "Baisley Park",
    11: "Bath Beach", 12: "Battery Park", 13: "Battery Park City",
    14: "Bay Ridge", 15: "Bay Terrace/Fort Totten", 16: "Bayside",
    17: "Bedford", 18: "Bedford Park", 19: "Bellerose", 20: "Belmont",
    21: "Bensonhurst East", 22: "Bensonhurst West",
    23: "Bloomfield/Emerson Hill", 24: "Bloomingdale", 25: "Boerum Hill",
    26: "Borough Park", 27: "Breezy Point/Riis Beach", 28: "Briarwood",
    29: "Brighton Beach", 30: "Broad Channel", 31: "Bronx Park",
    32: "Bronxdale", 33: "Brooklyn Heights", 34: "Brooklyn Navy Yard",
    35: "Brownsville", 36: "Bushwick North", 37: "Bushwick South",
    38: "Cambria Heights", 39: "Canarsie", 40: "Carroll Gardens",
    41: "Central Harlem", 42: "Central Harlem North", 43: "Central Park",
    44: "Charleston/Tottenville", 45: "Chinatown", 46: "City Island",
    47: "Clason Point", 48: "Clinton East", 49: "Clinton Hill",
    50: "Clinton West", 51: "Co-Op City", 52: "College Point",
    53: "Columbia St", 54: "Coney Island", 55: "Corona",
    56: "Country Club", 57: "Crotona Park", 58: "Crotona Park East",
    59: "Crown Heights North", 60: "Crown Heights South",
    61: "Cypress Hills", 62: "Douglaston",
    63: "Downtown Brooklyn/MetroTech", 64: "DUMBO/Vinegar Hill",
    65: "Dyker Heights", 66: "East Chelsea", 67: "East Concourse",
    68: "East Chelsea", 69: "East Elmhurst", 70: "East Flatbush/Farragut",
    71: "East Flatbush/Remsen Village", 72: "East Flushing",
    73: "East Harlem North", 74: "East Harlem South", 75: "East New York",
    76: "East New York/Penn Ave", 77: "East Tremont", 78: "East Village",
    79: "East Village", 80: "Eastchester", 81: "Elmhurst",
    82: "Elmhurst/Maspeth", 83: "Eltingville/Annadale",
    84: "Erasmus", 85: "Far Rockaway",
    86: "Financial District North", 87: "Financial District South",
    88: "Flatbush/Ditmas Park", 89: "Flatiron", 90: "Flatlands",
    91: "Flushing", 92: "Flushing Meadows-Corona Park",
    93: "Fordham South", 94: "Forest Hills", 95: "Forest Park",
    96: "Fort Greene", 97: "Fresh Meadows", 98: "Freshkills Park",
    99: "Garment District", 100: "Glen Oaks", 101: "Glendale",
    102: "Governor's Island", 103: "Gowanus", 104: "Gramercy",
    105: "Gravesend", 106: "Great Kills", 107: "Gramercy",
    108: "Green-Wood Cemetery", 109: "Greenpoint",
    110: "Greenwich Village North", 111: "Greenwich Village South",
    112: "Gowanus/Fort Greene", 113: "Hamilton Heights",
    114: "Hammels/Arverne", 115: "Heartland Village/Todt Hill",
    116: "Highbridge", 117: "Highbridge Park", 118: "Hillcrest/Pomonok",
    119: "Hollis", 120: "Homecrest", 121: "Howard Beach",
    122: "Hudson Sq", 123: "Hunts Point", 124: "Inwood",
    125: "Inwood Hill Park", 126: "Jackson Heights", 127: "Jamaica",
    128: "Jamaica Estates", 129: "JFK Airport", 130: "Kensington",
    131: "Kew Gardens", 132: "JFK Airport", 133: "Kew Gardens Hills",
    134: "Kingsbridge Heights", 135: "Kingsbridge/Marble Hill",
    136: "Kips Bay", 137: "LaGuardia Airport", 138: "LaGuardia Airport",
    139: "Laurelton", 140: "Lenox Hill East", 141: "Lenox Hill West",
    142: "Lincoln Square East", 143: "Lincoln Square West",
    144: "Little Italy/NoLiTa", 145: "Long Island City/Hunters Point",
    146: "Long Island City/Queens Plaza", 147: "Longwood",
    148: "Lower East Side", 149: "Madison", 150: "Manhattan Beach",
    151: "Manhattan Valley", 152: "Manhattanville",
    153: "Marble Hill", 154: "Marine Park/Floyd Bennett Field",
    155: "Marine Park/Mill Basin", 156: "Maspeth",
    157: "Meatpacking/West Village West", 158: "Melrose South",
    159: "Middle Village", 160: "Midtown Center", 161: "Midtown Center",
    162: "Midtown East", 163: "Midtown North", 164: "Midtown South",
    165: "Midwood", 166: "Morningside Heights",
    167: "Morrisania/Melrose", 168: "Mott Haven/Port Morris",
    169: "Mount Hope", 170: "Murray Hill", 171: "Murray Hill-Queens",
    172: "New Dorp/Midland Beach", 173: "Newark Airport",
    174: "North Corona", 175: "Norwood", 176: "Oakland Gardens",
    177: "Oakwood", 178: "Ocean Hill", 179: "Ocean Parkway South",
    180: "Old Astoria", 181: "Ozone Park", 182: "Park Slope",
    183: "Parkchester", 184: "Pelham Bay", 185: "Pelham Bay Park",
    186: "Penn Station/Madison Sq West", 187: "Pelham Pkwy",
    188: "Prospect-Lefferts Gardens", 189: "Prospect Heights",
    190: "Prospect Park", 191: "Queens Village",
    192: "Queensboro Hill", 193: "Queensbridge/Ravenswood",
    194: "Randalls Island", 195: "Red Hook", 196: "Rego Park",
    197: "Richmond Hill", 198: "Ridgewood", 199: "Rikers Island",
    200: "Riverdale/North Riverdale", 201: "Rockaway Park",
    202: "Rosedale", 203: "Rossville/Woodrow", 204: "Saint Albans",
    205: "Saint George/New Brighton",
    206: "Saint Michaels Cemetery/Woodside",
    207: "Schuylerville/Edgewater Park", 208: "Seaport",
    209: "Sheepshead Bay", 210: "SoHo", 211: "Soundview/Bruckner",
    212: "Soundview/Castle Hill", 213: "South Beach/Dongan Hills",
    214: "South Jamaica", 215: "South Ozone Park",
    216: "South Williamsburg", 217: "Springfield Gardens North",
    218: "Springfield Gardens South",
    219: "Spuyten Duyvil/Kingsbridge", 220: "Stapleton",
    221: "Starrett City", 222: "Steinway",
    223: "Stuyvesant Heights",
    224: "Stuyvesant Town/Peter Cooper Village",
    225: "Sunnyside", 226: "Sunset Park East", 227: "Sunset Park West",
    228: "Sutton Place/Turtle Bay North", 229: "Midwood",
    230: "Times Sq/Theatre District", 231: "TriBeCa/Civic Center",
    232: "Two Bridges/Seward Park", 233: "UN/Turtle Bay South",
    234: "Union Sq", 235: "University Heights/Morris Heights",
    236: "Upper East Side North", 237: "Upper East Side South",
    238: "Upper West Side North", 239: "Upper West Side South",
    240: "Van Cortlandt Park", 241: "Van Cortlandt Village",
    242: "Van Nest/Morris Park", 243: "Washington Heights North",
    244: "Washington Heights South", 245: "West Brighton",
    246: "West Concourse", 247: "West Farms/Bronx River",
    248: "West Village", 249: "West Village",
    250: "Westchester Village/Unionport", 251: "Westerleigh",
    252: "Whitestone", 253: "Willets Point",
    254: "Williamsbridge/Olinville", 255: "Williamsburg North",
    256: "Williamsburg South", 257: "Windsor Terrace",
    258: "Woodhaven", 259: "Woodlawn/Wakefield", 260: "Woodside",
    261: "World Trade Center", 262: "Yorkville East",
    263: "Yorkville West",
}

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def run(load_data):
    """Point d'entrée unique. `load_data` est la fonction (mise en cache par
    l'appelant) qui renvoie le tuple (daily, hourly, zones, profile)."""
    nyc_theme.inject_css()
    # Hors de st.sidebar : le CSS épingle ce bouton dans la bannière en
    # position fixe, et une barre repliée est déplacée par un `transform`, qui
    # ferait alors office de bloc conteneur et emporterait le bouton hors de
    # l'écran avec elle.
    nyc_theme.theme_toggle()

    daily, hourly, zones, profile, congestion, split, payment = _load(load_data)
    view = _sidebar()

    st.title("NYC Yellow Taxi")

    if view == PAGE_ABOUT:
        nyc_content.render_about_tab()
    elif view == PAGE_PIPELINE:
        nyc_content.render_pipeline_tab()
    else:
        _scope_strip(daily)
        section = view
        if section == SEC_HOW_MUCH:
            _tab_volumes(daily, congestion, split, payment)
        elif section == SEC_WHEN:
            _tab_patterns(daily, hourly)
        elif section == SEC_WHERE:
            _tab_geo(zones)
        else:
            _tab_trip(profile)

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


def _sidebar():
    """Les quatre sections du dashboard, puis les deux pages de contexte.

    Un seul `st.radio` et non deux listes empilées : deux widgets porteraient
    deux sélections indépendantes, et il faudrait arbitrer laquelle a changé
    en dernier. Ici l'état est unique, et la hiérarchie est affaire de CSS.
    """
    with st.sidebar:
        view = st.radio(
            "Vue", VIEWS, index=_index(VIEWS, "vue", DEFAULT_SECTION),
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


def _label(text):
    """Intitulé de section de sidebar — petites capitales espacées."""
    return (
        f"<div style='font-size:0.66rem; font-weight:600; letter-spacing:0.09em; "
        f"text-transform:uppercase; color:{nyc_theme.TEXT_DIM}; "
        f"margin-bottom:6px;'>{text}</div>"
    )


# ---------------------------------------------------------------------------
# Sous-onglet 1 — Vue d'ensemble
# ---------------------------------------------------------------------------
def _yoy_window(fd):
    """Fenêtres pour la comparaison annuelle.

    On compare l'année en cours (1er janvier -> dernier jour disponible) aux
    mêmes mois de l'année précédente. Les données s'arrêtant fin sept. 2025,
    cela oppose janv.-sept. 2025 à janv.-sept. 2024 : deux fenêtres de même
    longueur et de même saison. Comparer aux "12 mois précédents" aurait mis
    face à face des périodes de longueurs inégales, l'historique n'en couvrant
    que 21.
    """
    end = fd["PICKUP_DATE"].max()
    start = pd.Timestamp(year=end.year, month=1, day=1)
    cur = fd[(fd["PICKUP_DATE"] >= start) & (fd["PICKUP_DATE"] <= end)]
    prev = fd[(fd["PICKUP_DATE"] >= start - pd.DateOffset(years=1)) &
              (fd["PICKUP_DATE"] <= end - pd.DateOffset(years=1))]
    return cur, prev, start.year - 1


def _delta(cur, prev, col, how):
    """Variation en % entre les deux fenêtres, None si la précédente est vide."""
    if prev.empty or cur.empty:
        return None
    a = getattr(cur[col], how)()
    b = getattr(prev[col], how)()
    if not b:
        return None
    return (a - b) / b * 100


def _monthly(fd, col, how):
    """Série mensuelle pour la courbe miniature."""
    g = fd.set_index("PICKUP_DATE")[col].resample("MS")
    return (g.sum() if how == "sum" else g.mean()).dropna().tolist()


def _tab_volumes(fd, congestion, split, payment):
    card = nyc_theme.card
    cur, prev, prev_year = _yoy_window(fd)
    note = f"vs même période {prev_year}"

    st.header("Indicateurs clés")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card("Courses", nyc_theme.fmt_int(fd["TOTAL_TRIPS"].sum()), note,
             delta=_delta(cur, prev, "TOTAL_TRIPS", "sum"),
             spark=_monthly(fd, "TOTAL_TRIPS", "sum"))
    with c2:
        rev = fd["TOTAL_REVENUE"].sum()
        card("Revenus totaux", f"${rev/1e9:.2f}B",
             f"${rev/fd['TOTAL_TRIPS'].sum():.2f} / course &middot; {note}",
             delta=_delta(cur, prev, "TOTAL_REVENUE", "sum"),
             spark=_monthly(fd, "TOTAL_REVENUE", "sum"))
    with c3:
        card("Distance moyenne", f"{fd['AVG_DISTANCE'].mean():.1f} mi",
             f"par trajet &middot; {note}",
             delta=_delta(cur, prev, "AVG_DISTANCE", "mean"),
             spark=_monthly(fd, "AVG_DISTANCE", "mean"))
    with c4:
        card("Tarif moyen", f"${fd['AVG_FARE'].mean():.2f}",
             f"toutes charges incluses &middot; {note}",
             delta=_delta(cur, prev, "AVG_FARE", "mean"),
             spark=_monthly(fd, "AVG_FARE", "mean"))
    with c5:
        card("Pourboire moyen", f"{fd['AVG_TIP_PCT'].mean():.1f}%",
             f"du tarif de base &middot; {note}",
             delta=_delta(cur, prev, "AVG_TIP_PCT", "mean"),
             spark=_monthly(fd, "AVG_TIP_PCT", "mean"))

    st.divider()
    _congestion_block(congestion)
    st.divider()
    _revenue_split_block(split)
    st.divider()
    _tip_block(payment)


# Date d'entrée en vigueur du péage urbain de Manhattan (Congestion Relief Zone).
CBD_START = pd.Timestamp("2025-01-05")


def _congestion_block(congestion):
    """Péage urbain de Manhattan — une décision publique lisible dans la donnée.

    La colonne cbd_congestion_fee n'existe tout simplement pas dans les Parquet
    d'avant 2025 : la bascule n'est pas une tendance qu'on interprète, c'est un
    champ qui apparaît. D'où le choix d'une série quotidienne plutôt que
    mensuelle — à la journée, le palier est net.
    """
    st.header("Le péage urbain, vu depuis les compteurs")

    cg = congestion.copy()
    cg["PICKUP_DATE"] = pd.to_datetime(cg["PICKUP_DATE"], errors="coerce")
    cg = cg.dropna(subset=["PICKUP_DATE"]).sort_values("PICKUP_DATE")
    cg = cg[(cg["PICKUP_DATE"] >= "2024-01-01") & (cg["PICKUP_DATE"] <= "2025-09-30")]
    cg["PCT"] = cg["CBD_TRIPS"] / cg["TOTAL_TRIPS"] * 100

    after = cg[cg["PICKUP_DATE"] >= CBD_START]
    collected = after["CBD_COLLECTED"].sum()
    months = max(1, after["PICKUP_DATE"].dt.to_period("M").nunique())

    c1, c2, c3 = st.columns(3)
    with c1:
        nyc_theme.card("Entrée en vigueur", CBD_START.strftime("%d/%m/%Y"),
                       "première course taxée dans les données")
    with c2:
        nyc_theme.card("Courses concernées", f"{after['PCT'].mean():.0f}%",
                       "depuis la mise en service")
    with c3:
        nyc_theme.card("Perçu via les taxis jaunes", f"${collected/1e6:.1f}M",
                       f"soit ${collected/months/1e6:.1f}M par mois")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cg["PICKUP_DATE"], y=cg["PCT"], mode="lines",
        line=dict(color=nyc_theme.ACCENT, width=1.4), name="Courses taxées",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.1f} % des courses<extra></extra>",
    ))
    fig.add_vline(x=CBD_START.timestamp() * 1000, line_width=1,
                  line_dash="dot", line_color=nyc_theme.TEXT_DIM)
    fig.add_annotation(
        x=CBD_START, y=78, text="5 janv. 2025", showarrow=False,
        xanchor="left", xshift=6, font=dict(size=10, color=nyc_theme.TEXT_DIM),
    )
    nyc_theme.style_fig(fig)
    fig.update_layout(
        title="Part des courses soumises au péage urbain",
        height=300, showlegend=False,
        yaxis=dict(title="% des courses", range=[-3, 90]),
        xaxis_title="", margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Aucune course taxée jusqu'au 3 janvier 2025, 399 le 4, plus de 50 000 le 5. "
        "Le palier n'est pas un lissage statistique : la colonne `cbd_congestion_fee` "
        "est absente des fichiers antérieurs à 2025."
    )


def _revenue_split_block(split):
    """Décomposition du prix payé — le tarif moyen n'est pas un bloc."""
    st.header("Où va l'argent")

    r = split.iloc[0]
    trips = r["TOTAL_TRIPS"]
    parts = [
        ("Course au compteur", r["FARE"]),
        ("Pourboire", r["TIP"]),
        ("Surcharge congestion", r["CONGESTION_SURCHARGE"]),
        ("Péage urbain (CBD)", r["CBD_FEE"]),
        ("Péages routiers", r["TOLLS"]),
        ("Supplément aéroport", r["AIRPORT_FEE"]),
        ("Taxe MTA", r["MTA_TAX"]),
        ("Suppléments horaires", r["EXTRA"]),
        ("Surcharge amélioration", r["IMPROVEMENT"]),
    ]
    parts = [(k, v) for k, v in parts if v and v > 0]
    total = sum(v for _, v in parts)

    fig = go.Figure()
    for i, (name, value) in enumerate(parts):
        fig.add_trace(go.Bar(
            y=["prix"], x=[value / total * 100], name=name, orientation="h",
            marker_color=nyc_theme.ACCENT if i == 0 else f"rgba(139,135,148,{0.85 - i*0.07:.2f})",
            hovertemplate=f"<b>{name}</b><br>%{{x:.1f}} %<br>"
                          f"${value/trips:.2f} par course<extra></extra>",
        ))
    nyc_theme.style_fig(fig)
    fig.update_layout(
        barmode="stack", height=150,
        title=f"Composition du prix moyen (${total/trips:.2f} par course)",
        xaxis=dict(title="", ticksuffix=" %", range=[0, 100]),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="top", y=-0.35, font=dict(size=10)),
        margin=dict(t=40, b=10, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    rows = "".join(
        f'<div style="display:flex; justify-content:space-between; gap:12px; '
        f'padding:7px 0; border-bottom:1px solid {nyc_theme.RULE};">'
        f'<span style="color:{nyc_theme.TEXT_DIM}; font-size:0.8rem;">{name}</span>'
        f'<span style="font-variant-numeric:tabular-nums; font-size:0.8rem;">'
        f'<b style="color:{nyc_theme.TEXT_LIGHT};">${value/trips:.2f}</b>'
        f'<span style="color:{nyc_theme.TEXT_DIM};"> &middot; {value/total*100:.1f} %</span>'
        f'</span></div>'
        for name, value in parts
    )
    st.markdown(f"<div style='max-width:46ch'>{rows}</div>", unsafe_allow_html=True)


# Libellés officiels TLC du champ payment_type.
PAYMENT_LABELS = {
    1: "Carte bancaire", 2: "Espèces", 3: "Gratuit",
    4: "Litige", 5: "Inconnu", 6: "Course annulée", 0: "Non renseigné",
}


def _tip_block(payment):
    """Le taux de pourboire est une mesure de l'instrument, pas des passagers.

    Le compteur n'enregistre que les pourboires passés par la carte. Afficher
    un taux global sans le dire laisserait croire à un comportement.
    """
    st.header("Le pourboire mesure le terminal, pas la générosité")

    pay = payment.copy()
    pay["LABEL"] = pay["PAYMENT_TYPE"].map(PAYMENT_LABELS).fillna("Autre")
    pay["PART"] = pay["TOTAL_TRIPS"] / pay["TOTAL_TRIPS"].sum() * 100
    pay = pay[pay["PART"] >= 1].sort_values("TOTAL_TRIPS", ascending=False)

    header = (
        f'<div style="display:flex; gap:12px; padding:6px 0; border-bottom:1px solid '
        f'{nyc_theme.RULE}; font-size:0.66rem; letter-spacing:0.08em; '
        f'text-transform:uppercase; color:{nyc_theme.TEXT_DIM};">'
        f'<span style="flex:2;">Moyen de paiement</span>'
        f'<span style="flex:1; text-align:right;">Part des courses</span>'
        f'<span style="flex:1; text-align:right;">Avec pourboire</span></div>'
    )
    rows = "".join(
        f'<div style="display:flex; gap:12px; padding:9px 0; border-bottom:1px solid '
        f'{nyc_theme.RULE}; font-size:0.85rem; font-variant-numeric:tabular-nums;">'
        f'<span style="flex:2; color:{nyc_theme.TEXT_LIGHT};">{row.LABEL}</span>'
        f'<span style="flex:1; text-align:right; color:{nyc_theme.TEXT_DIM};">{row.PART:.1f} %</span>'
        f'<span style="flex:1; text-align:right; color:{nyc_theme.TEXT_LIGHT};">'
        f'{row.PCT_AVEC_POURBOIRE:.2f} %</span></div>'
        for row in pay.itertuples()
    )
    st.markdown(f"<div style='max-width:52ch'>{header}{rows}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{nyc_theme.TEXT_DIM}; font-size:0.82rem; line-height:1.65; "
        f"max-width:72ch; margin-top:14px;'>"
        "Les pourboires réglés en espèces ne transitent pas par le compteur : ils "
        "n'existent pas dans la donnée. Le taux affiché mesure donc le terminal de "
        "paiement, pas les passagers. Tout indicateur de pourboire calculé sur "
        "l'ensemble des courses est mécaniquement tiré vers le bas par la part payée "
        "en liquide.</p>",
        unsafe_allow_html=True,
    )


def _speed_block(hourly):
    """Vitesse moyenne par heure — la congestion mesurée, pas déduite.

    Calculée depuis tpep_dropoff_datetime, qui ne servait à rien jusqu'ici.
    L'intérêt est que la courbe ne se superpose pas au volume : le pic de
    courses est le soir, le creux de vitesse est en milieu d'après-midi.
    """
    if "AVG_SPEED_MPH" not in hourly.columns:
        return

    h = hourly.set_index("PICKUP_HOUR")["AVG_SPEED_MPH"].reindex(range(24))
    if h.isna().all():
        return

    fastest, slowest = int(h.idxmax()), int(h.idxmin())
    st.divider()
    st.header("La vitesse, autre lecture de la journée")

    c1, c2, c3 = st.columns(3)
    with c1:
        nyc_theme.card("Heure la plus fluide", f"{h[fastest]:.1f} mph", f"à {fastest}h")
    with c2:
        nyc_theme.card("Heure la plus lente", f"{h[slowest]:.1f} mph", f"à {slowest}h")
    with c3:
        nyc_theme.card("Écart", f"×{h[fastest]/h[slowest]:.1f}",
                       "entre le plus fluide et le plus lent")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(24)), y=h.tolist(), mode="lines",
        line=dict(color=nyc_theme.ACCENT, width=2),
        hovertemplate="%{x}h<br>%{y:.1f} mph<extra></extra>",
    ))
    for hour, color in ((fastest, nyc_theme.BETTER), (slowest, nyc_theme.WORSE)):
        fig.add_trace(go.Scatter(
            x=[hour], y=[h[hour]], mode="markers",
            marker=dict(color=color, size=7), hoverinfo="skip", showlegend=False,
        ))
    nyc_theme.style_fig(fig)
    fig.update_layout(
        title="Vitesse moyenne par heure de prise en charge",
        height=300, showlegend=False,
        yaxis=dict(title="mph"),
        xaxis=dict(title="", tickmode="array", tickvals=list(range(0, 24, 2)),
                   ticktext=[f"{x}h" for x in range(0, 24, 2)]),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Distance divisée par la durée réelle du trajet, courses de 1 min à 3 h. "
        "Le volume de courses culmine le soir, mais la circulation est au plus lent "
        "en milieu d'après-midi : les deux courbes ne racontent pas la même journée."
    )


def _tab_trip(profile):
    """Portrait type d'un trajet : ce que contient une course, pas combien.

    Section autonome depuis que le dashboard est découpé en questions :
    "Combien ?" porte les volumes, "Comment ?" porte la forme du trajet.
    """
    card = nyc_theme.card

    st.header("Portrait type d'un trajet")

    p = profile.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Distance moyenne", f"{p['AVG_DISTANCE']:.1f} mi",
             "par course")
    with c2:
        card("Tarif moyen", f"${p['AVG_FARE']:.2f}",
             "toutes charges incluses")
    with c3:
        card("Courses avec pourboire", f"{p['PCT_AVEC_POURBOIRE']:.0f}%",
             f"pourboire moy. {p['AVG_TIP_PCT']:.1f}% du tarif")
    with c4:
        card("Paiement par carte", f"{p['PCT_CARTE']:.0f}%",
             f"{100 - p['PCT_CARTE']:.0f}% en espèces")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Passagers / course", f"{p['AVG_PASSAGERS']:.1f}",
             "en moyenne")
    with c2:
        card("Courses via JFK", f"{p['PCT_AEROPORT_JFK']:.1f}%",
             "départ ou arrivée")
    with c3:
        card("Courses via LaGuardia", f"{p['PCT_AEROPORT_LGA']:.1f}%",
             "départ ou arrivée")
    with c4:
        pct_city = 100 - p['PCT_AEROPORT_JFK'] - p['PCT_AEROPORT_LGA']
        card("Courses intra-ville", f"{pct_city:.0f}%",
             "sans aéroport")

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini chart : répartition paiement
    pay_data = pd.DataFrame({
        "Mode": ["Carte bancaire", "Espèces", "Autre"],
        "Part (%)": [p["PCT_CARTE"], 100 - p["PCT_CARTE"] - 3, 3],
    })
    fig_pay = px.bar(
        pay_data,
        x="Part (%)",
        y="Mode",
        orientation="h",
        color="Mode",
        color_discrete_sequence=[nyc_theme.TAXI_GOLD, nyc_theme.DOLLAR_GREEN, nyc_theme.TEXT_DIM],
        title="Répartition des modes de paiement",
        text="Part (%)",
    )
    fig_pay.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
    nyc_theme.style_fig(fig_pay)
    fig_pay.update_layout(
        height=180, showlegend=False,
        xaxis=dict(range=[0, 100], title=""),
        yaxis_title="",
        margin=dict(l=0, t=40, b=10),
    )
    _, col_pay, _ = st.columns([0.25, 0.5, 0.25])
    with col_pay:
        st.plotly_chart(fig_pay, use_container_width=True)

# ---------------------------------------------------------------------------
# Sous-onglet 2 — Temporalité
# ---------------------------------------------------------------------------

def _tab_patterns(fd, hourly):
    st.header("Patterns d'activité")

    METRIC_OPTIONS = ["TOTAL_TRIPS", "TOTAL_REVENUE", "AVG_FARE", "AVG_TIP_PCT", "AVG_DISTANCE"]
    METRIC_LABELS  = {
        "TOTAL_TRIPS":   "Nombre de courses",
        "TOTAL_REVENUE": "Revenus ($)",
        "AVG_FARE":      "Tarif moyen ($)",
        "AVG_TIP_PCT":   "Pourboire moyen (%)",
        "AVG_DISTANCE":  "Distance moyenne (mi)",
    }
    METRIC_NAMES = {
        "TOTAL_TRIPS":   "Nb de courses",
        "TOTAL_REVENUE": "Revenus ($)",
        "AVG_FARE":      "Tarif moyen ($)",
        "AVG_TIP_PCT":   "Pourboire (%)",
        "AVG_DISTANCE":  "Distance (mi)",
    }

    # Valeur courante depuis session_state (persiste entre les reruns)
    metric_choice = st.session_state.get("metric_pills", "TOTAL_TRIPS")
    metric_label  = METRIC_LABELS[metric_choice]

    col1, col2 = st.columns(2)

    with col1:
        fd_copy = fd.copy()
        fd_copy["jour_semaine"] = fd_copy["PICKUP_DATE"].dt.day_name()  # type: ignore
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_fr    = ["Lun.", "Mar.", "Mer.", "Jeu.", "Ven.", "Sam.", "Dim."]

        weekly = (
            fd_copy.groupby("jour_semaine")[metric_choice]  # type: ignore
            .mean()
            .reindex(day_order)
            .reset_index()
        )
        weekly["jour_fr"] = day_fr
        mean_val = weekly[metric_choice].mean()
        weekly["color"] = weekly[metric_choice].apply(
            lambda v: nyc_theme.ACCENT if v >= mean_val else nyc_theme.BAR_MUTED
        )
        weekly["delta"] = ((weekly[metric_choice] - mean_val) / mean_val * 100).round(1)

        fig_week = go.Figure()
        fig_week.add_trace(go.Bar(
            x=weekly["jour_fr"],
            y=weekly[metric_choice],
            marker_color=weekly["color"],
            text=weekly["delta"].apply(lambda d: f"+{d}%" if d >= 0 else f"{d}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + metric_label + " : %{y:,.0f}<extra></extra>",
        ))
        fig_week.add_hline(
            y=mean_val,
            line_dash="dot",
            line_color=nyc_theme.TEXT_DIM,
            annotation_text="moy.",
            annotation_position="right",
            annotation_font_size=11,
        )
        y_min = weekly[metric_choice].min()
        y_max = weekly[metric_choice].max()
        nyc_theme.style_fig(fig_week)
        fig_week.update_layout(
            title=f"{metric_label} — profil hebdomadaire",
            height=380,
            showlegend=False,
            yaxis_range=[y_min * 0.96, y_max * 1.08],
            yaxis_title=metric_label,
            xaxis_title="",
        )
        st.plotly_chart(fig_week, use_container_width=True)

    with col2:
        hourly_vals = (
            hourly.set_index("PICKUP_HOUR")[metric_choice]
            .reindex(range(24), fill_value=0)
        )
        values_by_hour = hourly_vals.tolist()
        peak_h   = int(hourly_vals.idxmax())
        trough_h = int(hourly_vals.idxmin())

        def _fmt_metric(v):
            if metric_choice == "TOTAL_REVENUE":
                return f"${v/1e3:,.0f}k".replace(",", " ")
            if metric_choice == "AVG_FARE":
                return f"${v:,.2f}"
            if metric_choice == "AVG_TIP_PCT":
                return f"{v:.1f}%"
            if metric_choice == "AVG_DISTANCE":
                return f"{v:.1f} mi"
            return f"{v:,.0f}".replace(",", " ")

        # Barres horaires : seuls le pic et le creux sont accentués, le reste
        # reste neutre — on lit la forme de la journée, pas 24 couleurs.
        bar_colors = [
            nyc_theme.BETTER if h == peak_h
            else nyc_theme.WORSE if h == trough_h
            else nyc_theme.BAR_MUTED
            for h in range(24)
        ]
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Bar(
            x=list(range(24)),
            y=values_by_hour,
            marker_color=bar_colors,
            hovertemplate="%{x}h<br>" + metric_label + " : %{y:,.1f}<extra></extra>",
        ))
        for h, anchor in ((peak_h, "bottom"), (trough_h, "bottom")):
            if h == trough_h and trough_h == peak_h:
                continue
            fig_hour.add_annotation(
                x=h, y=values_by_hour[h], text=_fmt_metric(values_by_hour[h]),
                showarrow=False, yanchor=anchor, yshift=6,
                font=dict(size=10, color=nyc_theme.TEXT_DIM),
            )
        nyc_theme.style_fig(fig_hour)
        fig_hour.update_layout(
            title=f"{metric_label} — profil horaire",
            height=380,
            showlegend=False,
            bargap=0.25,
            yaxis_title=metric_label,
            xaxis=dict(title="", tickmode="array",
                       tickvals=list(range(0, 24, 2)),
                       ticktext=[f"{h}h" for h in range(0, 24, 2)]),
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        st.caption(
            (
                f"Pic à {peak_h}h ({_fmt_metric(values_by_hour[peak_h])}) · "
                f"creux à {trough_h}h ({_fmt_metric(values_by_hour[trough_h])})"
            ).replace("$", r"\$")  # évite l'interprétation LaTeX ($...$) de st.caption
        )

    _speed_block(hourly)

    # Sélecteur de métrique — boutons natifs centrés entre les deux rangées
    st.markdown(
        "<p style='text-align:center; font-size:0.72rem; font-weight:600; "
        "letter-spacing:0.09em; text-transform:uppercase; "
        f"color:{nyc_theme.TEXT_DIM}; margin:22px 0 8px;'>"
        "Métrique affichée</p>",
        unsafe_allow_html=True,
    )
    # Pleine largeur : avec la sidebar, une gouttière latérale rognait assez les
    # boutons pour couper leurs libellés.
    cols = st.columns(len(METRIC_OPTIONS))
    for col, opt in zip(cols, METRIC_OPTIONS):
        with col:
            btn_type = "primary" if opt == metric_choice else "secondary"
            if st.button(METRIC_NAMES[opt], key=f"m_{opt}",
                         type=btn_type, use_container_width=True):
                st.session_state["metric_pills"] = opt
                st.rerun()
    metric_choice = st.session_state.get("metric_pills", metric_choice)
    metric_label  = METRIC_LABELS[metric_choice]
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -- Évolution pleine largeur ------------------------------------------
    fd_sorted = fd.sort_values("PICKUP_DATE").copy()  # type: ignore
    fd_sorted["MA7"] = fd_sorted[metric_choice].rolling(7, center=True, min_periods=1).mean()

    date_global_min = fd_sorted["PICKUP_DATE"].min()
    date_global_max = fd_sorted["PICKUP_DATE"].max()

    PERIODS = {"1M": 1, "3M": 3, "6M": 6, "1A": 12, "Tout": 0}
    if "evo_period" not in st.session_state:
        st.session_state["evo_period"] = "6M"
    if "evo_offset" not in st.session_state:
        st.session_state["evo_offset"] = 0

    cur_period = st.session_state["evo_period"]
    cur_offset = st.session_state["evo_offset"]

    if cur_period == "Tout":
        x_min, x_max = date_global_min, date_global_max
    else:
        months = PERIODS[cur_period]
        x_max = date_global_max - pd.DateOffset(months=cur_offset * months)
        x_min = x_max - pd.DateOffset(months=months)
        x_min = max(x_min, date_global_min)
        x_max = min(x_max, date_global_max)

    # Contrôles période — radio discret + ← → à droite
    ctl_l, ctl_r = st.columns([6, 1])
    with ctl_l:
        sel_period = st.radio(
            "Période", options=list(PERIODS.keys()),
            index=list(PERIODS.keys()).index(cur_period),
            horizontal=True, key="evo_radio",
            label_visibility="collapsed",
        )
        if sel_period != cur_period:
            st.session_state["evo_period"] = sel_period
            st.session_state["evo_offset"] = 0
            st.rerun()
    with ctl_r:
        nav_l, nav_r = st.columns(2)
        can_prev = cur_period != "Tout" and x_min > date_global_min
        can_next = cur_period != "Tout" and cur_offset > 0
        with nav_l:
            if st.button("←", key="evo_prev", disabled=not can_prev,
                         use_container_width=True):
                st.session_state["evo_offset"] += 1
                st.rerun()
        with nav_r:
            if st.button("→", key="evo_next", disabled=not can_next,
                         use_container_width=True):
                st.session_state["evo_offset"] -= 1
                st.rerun()

    # Min/max sur la fenêtre visible
    fd_vis = fd_sorted[
        (fd_sorted["PICKUP_DATE"] >= x_min) &
        (fd_sorted["PICKUP_DATE"] <= x_max)
    ]
    ann_max = ann_min = None
    if not fd_vis.empty:
        idx_max = fd_vis[metric_choice].idxmax()  # type: ignore
        idx_min = fd_vis[metric_choice].idxmin()  # type: ignore
        ann_max = dict(
            x=fd_vis.loc[idx_max, "PICKUP_DATE"], y=fd_vis.loc[idx_max, metric_choice],
            text=f"▲ {fd_vis.loc[idx_max, 'PICKUP_DATE'].strftime('%-d %b %Y')}",
            showarrow=True, arrowhead=2, arrowcolor=nyc_theme.BETTER,
            font=dict(size=10, color=nyc_theme.BETTER),
            bgcolor=nyc_theme.BG_PANEL, bordercolor=nyc_theme.BETTER,
            borderwidth=1, borderpad=3, ax=0, ay=-36,
        )
        ann_min = dict(
            x=fd_vis.loc[idx_min, "PICKUP_DATE"], y=fd_vis.loc[idx_min, metric_choice],
            text=f"▼ {fd_vis.loc[idx_min, 'PICKUP_DATE'].strftime('%-d %b %Y')}",
            showarrow=True, arrowhead=2, arrowcolor=nyc_theme.WORSE,
            font=dict(size=10, color=nyc_theme.WORSE),
            bgcolor=nyc_theme.BG_PANEL, bordercolor=nyc_theme.WORSE,
            borderwidth=1, borderpad=3, ax=0, ay=36,
        )

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=fd_sorted["PICKUP_DATE"], y=fd_sorted[metric_choice],
        mode="lines", line=dict(color=nyc_theme.BAR_MUTED, width=1), name="Quotidien",
        hovertemplate="%{x|%d %b %Y}<br>" + metric_label + " : %{y:,.1f}<extra></extra>",
    ))
    fig_line.add_trace(go.Scatter(
        x=fd_sorted["PICKUP_DATE"], y=fd_sorted["MA7"],
        mode="lines", line=dict(color=nyc_theme.TAXI_GOLD, width=2.5), name="Moy. 7 jours",
        hovertemplate="%{x|%d %b %Y}<br>Moy. 7j : %{y:,.1f}<extra></extra>",
    ))
    if ann_max:
        fig_line.add_annotation(**ann_max)
    if ann_min:
        fig_line.add_annotation(**ann_min)
    def bar_rank(data, title, color, height):
        data = data.copy()
        data["date_fr"] = data["PICKUP_DATE"].dt.strftime("%-d %b %Y")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data[metric_choice], y=data["date_fr"], orientation="h",
            marker=dict(color=color),
            hovertemplate="<b>%{y}</b><br>" + metric_label + " : %{x:,.1f}<extra></extra>",
        ))
        nyc_theme.style_fig(fig)
        fig.update_layout(
            title=title, height=height,
            xaxis_title="", yaxis_title="",
            margin=dict(l=0, t=36, b=4, r=8),
            font=dict(size=10, color=nyc_theme.TEXT_LIGHT),
        )
        return fig

    top10 = fd.nlargest(10, metric_choice).sort_values(metric_choice)  # type: ignore
    bot10 = fd.nsmallest(10, metric_choice).sort_values(metric_choice, ascending=False)  # type: ignore

    evo_col, rank_col = st.columns(2)

    with evo_col:
        nyc_theme.style_fig(fig_line)
        fig_line.update_layout(
            title=f"{metric_label} — évolution quotidienne",
            height=420,
            yaxis_title=metric_label, xaxis_title="",
            xaxis=dict(range=[x_min, x_max], autorange=False),
            # Légende sous le graphique : au-dessus elle chevauchait le titre.
            legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="right", x=1),
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with rank_col:
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(bar_rank(top10, "10 meilleurs jours",
                                     nyc_theme.BETTER, 420),
                            use_container_width=True)
        with r2:
            st.plotly_chart(bar_rank(bot10, "10 pires jours",
                                     nyc_theme.WORSE, 420),
                            use_container_width=True)

# ---------------------------------------------------------------------------
# Sous-onglet 3 — Géographie
# ---------------------------------------------------------------------------
def _tab_geo(zones):
    st.header("Quartiers")

    zones["zone_name"] = zones["ZONE_ID"].map(ZONE_LOOKUP).fillna(zones["ZONE_ID"].astype(str))  # type: ignore

    fig_tree = px.treemap(
        zones,
        path=[px.Constant("NYC"), "zone_name"],
        values="TOTAL_TRIPS",
        color="AVG_FARE",
        color_continuous_scale=[[0.0, "#3a3650"], [0.5, "#9f948a"], [1.0, nyc_theme.TAXI_GOLD]],
        template="plotly_dark",
    )
    fig_tree.update_traces(
        root_color=nyc_theme.BG_PAGE,
        # pathbar masquée : le fil d'Ariane du treemap affichait "undefined"
        # au-dessus du graphique (racine px.Constant sans libellé propre).
        pathbar=dict(visible=False),
        hovertemplate="<b>%{label}</b><br>Courses : %{value:,.0f}<extra></extra>",
    )
    nyc_theme.style_fig(fig_tree)
    fig_tree.update_layout(height=500, margin=dict(t=10, l=0, r=0, b=0))
    fig_tree.update_coloraxes(colorbar=dict(title="Tarif moy. ($)", thickness=12, len=0.6))
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown(
        "**100 zones affichées sur ~260 zones TLC NYC · "
        "surface = volume de courses · couleur = tarif moyen (doré = plus cher)**"
    )
