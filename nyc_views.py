"""Les quatre sections du dashboard — un graphique par idée, pas un bloc par colonne.

Chaque section tient sur une page et porte quatre graphiques au plus. Les
indicateurs isolés sont réduits à trois par section : un nombre sans
comparaison n'apprend rien, et quinze cartes empilées se lisent moins bien que
quatre figures qui disent chacune une chose.

Découpage :

    Combien ?   ce que la course rapporte, et à qui
    Quand ?     les deux cycles de la ville, semaine et année
    Où ?        d'où l'on part, où l'on arrive, et ce qui ne revient pas
    Comment ?   la forme du trajet et ce que la source sait vraiment

Les tables croisées (heure × jour, flux entre arrondissements, tranches de
distance) sont produites hors ligne par scripts/G_build_aggregates.py. Les
anciennes tables à une dimension ne pouvaient porter qu'un classement ; les
croisements sont ce qui autorise une heatmap, un Sankey et un quadrant.

Choix de représentation écartés volontairement, et pourquoi :

  * barres empilées pour la composition du prix -> waterfall. Dans un
    empilement, seul le premier segment part d'une base commune ; les suivants
    flottent et ne se comparent plus. Le waterfall garde la base au sol.
  * treemap des zones -> barres classées. Le treemap encodait le volume par
    l'aire et le tarif par la couleur : l'œil compare mal des aires, et la
    double lecture faisait passer deux messages pour un.
  * axe tronqué sur le profil hebdomadaire. Une barre dit « combien » par sa
    hauteur ; couper l'axe à 96 % du minimum multipliait visuellement un écart
    de 8 % par trois.
  * double axe volume / vitesse. Deux échelles sur un même cadre laissent
    choisir où les courbes se croisent. Les deux lectures sont séparées.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nyc_theme
import nyc_zones

# dayofweek de DuckDB : 0 = dimanche. L'affichage commence au lundi.
DOW_ORDER = [1, 2, 3, 4, 5, 6, 0]
DOW_LABELS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

# Libellés officiels TLC du champ payment_type.
PAYMENT_LABELS = {
    1: "Carte bancaire", 2: "Espèces", 3: "Gratuit",
    4: "Litige", 5: "Inconnu", 6: "Course annulée", 0: "Non renseigné",
}

# Les mois en toutes lettres : strftime suit la locale du système, qui n'est
# pas celle du visiteur. Une liste évite de dépendre d'un `setlocale`.
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
          "août", "septembre", "octobre", "novembre", "décembre"]

# Date d'entrée en vigueur du péage urbain de Manhattan.
CBD_START = pd.Timestamp("2025-01-05")


# ---------------------------------------------------------------------------
# Outils communs
# ---------------------------------------------------------------------------
def _missing(name):
    """Avis quand une table croisée n'a pas encore été construite.

    Les six tables croisées ne sont pas calculées au démarrage : elles
    demandent une relecture des 74 M de lignes brutes. Une section qui en
    dépend le dit au lieu de lever une exception.
    """
    st.info(
        f"Table `{name}` absente. Construire les agrégats croisés :\n\n"
        "```\npython scripts/G_build_aggregates.py\n```"
    )


def _title(text, note=""):
    """Titre de bloc, avec une ligne de lecture sous le titre.

    La phrase vit au-dessus du graphique et non en légende : elle dit ce
    qu'il faut y voir, et elle doit être lue avant, pas après.
    """
    p = nyc_theme.palette()
    sub = (
        f"<div style='color:{p['TEXT_DIM']}; font-size:0.84rem; line-height:1.55; "
        f"max-width:80ch; margin:-2px 0 10px;'>{note}</div>" if note else ""
    )
    st.markdown(
        f"<h3 style='font-size:1.02rem; font-weight:600; letter-spacing:-0.01em; "
        f"margin:26px 0 6px;'>{text}</h3>{sub}",
        unsafe_allow_html=True,
    )


def _fig(fig, height=340, **layout):
    """Applique le thème, pose les marges par défaut, affiche.

    Les marges sont un `setdefault` et non une valeur fixe : une figure qui
    porte une légende au-dessus du cadre a besoin de place en haut, et la
    passer en argument écraserait sinon la valeur par défaut deux fois.
    """
    layout.setdefault("margin", dict(t=10, b=10, l=0, r=0))
    nyc_theme.style_fig(fig)
    fig.update_layout(height=height, **layout)
    st.plotly_chart(fig, use_container_width=True)


def _rgba(hex_color, alpha):
    """Même couleur, rendue translucide — pour les rubans d'un Sankey.

    Un ruban opaque masque ceux qu'il croise ; l'alpha ne peut pas se noter
    dans un hexadécimal lu depuis config.toml.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# Rampe séquentielle construite autour de l'accent doré. Une heatmap réclame
# une progression monotone en clarté : partir du fond du thème et finir sur
# l'accent donne des cases hautes trop proches des moyennes. Les quatre arrêts
# sont posés à la main, en clarté décroissante côté sombre et croissante côté
# clair, pour que l'ordre des valeurs se lise sans la barre de couleur.
_HEAT = {
    "light": ["#faf7f0", "#ecdcb0", "#dcb45c", "#c08a1e", "#7d5410"],
    "dark": ["#14141b", "#3d3419", "#7a6124", "#c3912f", "#f0c558"],
}


def _heat():
    stops = _HEAT[nyc_theme.mode()]
    return [[i / (len(stops) - 1), c] for i, c in enumerate(stops)]


def _known(od):
    """Flux dont les deux extrémités sont un arrondissement identifié.

    Le référentiel TLC contient deux identifiants fourre-tout, 264 et 265,
    dont l'arrondissement vaut « Unknown » ou « N/A ». Ils représentent des
    courses dont la zone n'a pas été relevée ; les garder ferait apparaître
    un septième arrondissement dans un Sankey.
    """
    valid = set(nyc_zones.BOROUGHS)
    return od[od["PU_BOROUGH"].isin(valid) & od["DO_BOROUGH"].isin(valid)]


def _fr(value, digits=1, unit=""):
    """Nombre décimal écrit à la française, virgule comprise."""
    return f"{value:.{digits}f}".replace(".", ",") + unit


def _dollars(v):
    """$2.08B, $312M, $45k — une seule unité, choisie par l'ordre de grandeur."""
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"${v / div:.2f}{suffix}" if div == 1e9 else f"${v / div:.0f}{suffix}"
    return f"${v:.0f}"


# ===========================================================================
# COMBIEN ? — ce que la course rapporte
# ===========================================================================
def how_much(daily, split, od, congestion):
    p = nyc_theme.palette()
    r = split.iloc[0]
    trips = r["TOTAL_TRIPS"]

    st.header("Ce que rapporte une course")

    c1, c2, c3 = st.columns(3)
    with c1:
        nyc_theme.card("Encaissé sur la période", _dollars(r["TOTAL"]),
                       f"{nyc_theme.fmt_int(trips)} courses")
    with c2:
        nyc_theme.card("Par course", f"${r['TOTAL'] / trips:.2f}",
                       "tout compris, pourboire et taxes")
    with c3:
        nyc_theme.card("Part du compteur", f"{r['FARE'] / r['TOTAL'] * 100:.0f} %",
                       "le reste est pourboire, taxes et péages")

    # -- 1. Waterfall de la composition du prix ----------------------------
    parts = [
        ("Compteur", r["FARE"]),
        ("Pourboire", r["TIP"]),
        ("Suppl. horaires", r["EXTRA"]),
        ("Surcharge congestion", r["CONGESTION_SURCHARGE"]),
        ("Péage urbain", r["CBD_FEE"]),
        ("Péages routiers", r["TOLLS"]),
        ("Taxe MTA", r["MTA_TAX"]),
        ("Suppl. aéroport", r["AIRPORT_FEE"]),
        ("Surcharge amélioration", r["IMPROVEMENT"]),
    ]
    parts = [(k, v) for k, v in parts if v and v > 0]

    _title(
        "Répartition de la somme totale payée",
        "À lire de gauche à droite : chaque marche ajoute une ligne de facturation "
        "au prix de la course précédente, jusqu'au montant débité. Le compteur pose "
        f"${r['FARE'] / trips:.2f}, les {len(parts) - 1} lignes suivantes en ajoutent "
        f"${(r['TOTAL'] - r['FARE']) / trips:.2f} — un tiers du total que le tarif "
        "affiché ne laisse pas voir.",
    )

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(parts) + ["total"],
        x=[k for k, _ in parts] + ["Total payé"],
        y=[v / trips for _, v in parts] + [0],
        connector=dict(line=dict(color=p["RULE"], width=1)),
        increasing=dict(marker=dict(color=p["BAR_MUTED"])),
        totals=dict(marker=dict(color=p["ACCENT"])),
        text=[f"${v / trips:.2f}" for _, v in parts] + [f"${r['TOTAL'] / trips:.2f}"],
        textposition="outside",
        textfont=dict(size=10, color=p["TEXT_DIM"]),
        hovertemplate="<b>%{x}</b><br>$%{y:.2f} par course<extra></extra>",
    ))
    _fig(fig, 360, yaxis=dict(title="$ par course", rangemode="tozero"),
         xaxis=dict(tickangle=-30), showlegend=False)

    # -- 2. Indices base 100 : volume, revenu, prix ------------------------
    _title(
        "Le marché grossit, le prix ne bouge pas",
        "Base 100 : la moyenne mensuelle de 2024. Les trois séries sont ramenées à "
        "une même base plutôt que posées sur deux axes — avec deux échelles, on "
        "choisit où les courbes se croisent.",
    )
    m = daily.set_index("PICKUP_DATE").resample("MS").agg(
        TRIPS=("TOTAL_TRIPS", "sum"),
        REVENUE=("TOTAL_REVENUE", "sum"),
    )
    m["PRICE"] = m["REVENUE"] / m["TRIPS"]
    # La base est la moyenne de 2024 et non le premier mois : janvier est le
    # mois le plus creux de l'année, et partir de lui ferait lire la
    # saisonnalité comme une croissance.
    idx = m / m[m.index.year == 2024].mean() * 100

    series = (
        ("TRIPS", "Courses", p["BAR_MUTED"], 1.6),
        ("REVENUE", "Recettes", p["ACCENT"], 2.6),
        ("PRICE", "Prix moyen", p["BETTER"], 1.6),
    )
    fig = go.Figure()
    for col, name, color, width in series:
        fig.add_trace(go.Scatter(
            x=idx.index, y=idx[col], mode="lines", name=name,
            line=dict(color=color, width=width),
            hovertemplate="%{x|%b %Y}<br>" + name + " : indice %{y:.1f}<extra></extra>",
        ))
    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color=p["TEXT_DIM"])
    # Étiquette en bout de courbe plutôt qu'une légende : le nom se lit à
    # l'endroit où l'œil finit la ligne. Deux séries qui terminent au même
    # niveau se chevauchent, d'où le décalage appliqué de proche en proche.
    last = idx.iloc[-1]
    placed = []
    for col, name, color, _w in sorted(series, key=lambda c: -last[c[0]]):
        shift = 0
        while any(abs(last[col] - v) < 6 and s_ == shift for v, s_ in placed):
            shift -= 12
        placed.append((last[col], shift))
        fig.add_annotation(
            x=idx.index[-1], y=last[col], text=f" {name} {last[col]:.0f}",
            showarrow=False, xanchor="left", yshift=shift,
            font=dict(size=10, color=color),
        )
    _fig(fig, 330, showlegend=False,
         yaxis=dict(title="indice, moyenne 2024 = 100"),
         xaxis=dict(title="",
                    range=[idx.index[0], idx.index[-1] + pd.DateOffset(months=3)]))
    growth = m["TRIPS"].iloc[-1] / m["TRIPS"].iloc[0] - 1
    price_growth = m["PRICE"].iloc[-1] / m["PRICE"].iloc[0] - 1
    st.caption(
        f"De janvier 2024 à {MONTHS[m.index[-1].month - 1]} {m.index[-1].year}, les "
        f"courses gagnent {growth * 100:.0f} % et le prix moyen "
        f"{price_growth * 100:+.0f} %. La croissance vient "
        "du volume, pas du tarif. Le décrochage de janvier 2025 suit l'entrée en "
        "vigueur du péage urbain, qui déplace une partie des courses hors de la "
        "zone taxée."
    )

    # -- 3. Quadrant volume / valeur par flux ------------------------------
    if od is None:
        _missing("od_borough")
    else:
        _title(
            "Deux manières de gagner de l'argent",
            "Chaque bulle est un flux entre deux arrondissements. À droite, le "
            "volume ; en haut, la recette par course. Manhattan interne vit du "
            "nombre, les liaisons aéroport de la valeur unitaire.",
        )
        f = _known(od)
        f = f[f["TOTAL_TRIPS"] >= 20_000].copy()
        f["PER_TRIP"] = f["TOTAL_REVENUE"] / f["TOTAL_TRIPS"]
        f["LABEL"] = f["PU_BOROUGH"] + " → " + f["DO_BOROUGH"]
        # Étiquettes réservées aux flux qui portent le propos : les plus gros
        # et les mieux payés. Vingt-cinq libellés se chevauchaient.
        shown = set(f.nlargest(4, "TOTAL_REVENUE")["LABEL"]) | set(
            f.nlargest(4, "PER_TRIP")["LABEL"])
        f["TEXT"] = [lab if lab in shown else "" for lab in f["LABEL"]]

        fig = go.Figure()
        for borough, g in f.groupby("PU_BOROUGH"):
            fig.add_trace(go.Scatter(
                x=g["TOTAL_TRIPS"], y=g["PER_TRIP"], mode="markers+text",
                name=borough, text=g["TEXT"], textposition="top center",
                textfont=dict(size=9, color=p["TEXT_DIM"]),
                customdata=np.c_[g["TOTAL_REVENUE"] / 1e6, g["AVG_DISTANCE"],
                                 g["LABEL"]],
                marker=dict(
                    # Aire proportionnelle aux recettes : `sizeref` cale la
                    # plus grosse bulle sur 64 px, les autres suivent.
                    size=g["TOTAL_REVENUE"],
                    sizemode="area", sizemin=4,
                    sizeref=f["TOTAL_REVENUE"].max() / 64 ** 2 * 2,
                    color=nyc_zones.color(borough),
                    line=dict(width=0.5, color=p["BG_PAGE"]),
                    opacity=0.85,
                ),
                hovertemplate="<b>%{customdata[2]}</b><br>%{x:,.0f} courses<br>"
                              "$%{y:.2f} par course<br>%{customdata[0]:.0f} M$ au total"
                              "<br>%{customdata[1]:.1f} mi en moyenne<extra></extra>",
            ))
        _fig(fig, 420, showlegend=False,
             xaxis=dict(title="courses sur la période (échelle log)", type="log"),
             yaxis=dict(title="$ par course", rangemode="tozero"),
             margin=dict(t=10, b=10, l=0, r=20))
        st.caption(
            "Aire des bulles proportionnelle aux recettes du flux. Flux de moins de "
            "20 000 courses écartés : à cette échelle, ils ne portent plus de moyenne "
            "fiable."
        )

    # -- 4. Péage urbain ---------------------------------------------------
    cg = congestion.copy()
    cg["PICKUP_DATE"] = pd.to_datetime(cg["PICKUP_DATE"], errors="coerce")
    cg = cg.dropna(subset=["PICKUP_DATE"]).sort_values("PICKUP_DATE")
    cg = cg[(cg["PICKUP_DATE"] >= "2024-01-01") & (cg["PICKUP_DATE"] <= "2025-09-30")]
    cg["PCT"] = cg["CBD_TRIPS"] / cg["TOTAL_TRIPS"] * 100
    after = cg[cg["PICKUP_DATE"] >= CBD_START]
    collected = after["CBD_COLLECTED"].sum()
    months = max(1, after["PICKUP_DATE"].dt.to_period("M").nunique())

    _title(
        "Une décision publique, visible au jour près",
        f"Le péage urbain de Manhattan entre en vigueur le 5 janvier 2025. "
        f"{_dollars(collected)} perçus depuis, soit {_dollars(collected / months)} "
        f"par mois sur les seuls taxis jaunes.",
    )
    fig = go.Figure(go.Scatter(
        x=cg["PICKUP_DATE"], y=cg["PCT"], mode="lines",
        line=dict(color=p["ACCENT"], width=1.3),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.1f} % des courses<extra></extra>",
    ))
    fig.add_vline(x=CBD_START.timestamp() * 1000, line_width=1,
                  line_dash="dot", line_color=p["TEXT_DIM"])
    fig.add_annotation(x=CBD_START, y=80, text="5 janv. 2025", showarrow=False,
                       xanchor="left", xshift=6,
                       font=dict(size=10, color=p["TEXT_DIM"]))
    _fig(fig, 280, showlegend=False,
         yaxis=dict(title="% des courses taxées", range=[-3, 92]),
         xaxis=dict(title=""))
    st.caption(
        "Aucune course taxée jusqu'au 3 janvier 2025, 399 le 4, plus de 50 000 le 5. "
        "Le palier n'est pas un lissage : la colonne `cbd_congestion_fee` est absente "
        "des fichiers antérieurs à 2025."
    )


# ===========================================================================
# QUAND ? — les deux cycles de la ville
# ===========================================================================
def when(daily, hw):
    p = nyc_theme.palette()

    st.header("Les deux horloges de la ville")

    if hw is None:
        _missing("hour_weekday")
    else:
        grid = hw.pivot(index="DOW", columns="PICKUP_HOUR", values="TOTAL_TRIPS")
        grid = grid.reindex(DOW_ORDER)
        peak = hw.loc[hw["TOTAL_TRIPS"].idxmax()]
        low = hw.loc[hw["TOTAL_TRIPS"].idxmin()]

        c1, c2, c3 = st.columns(3)
        with c1:
            nyc_theme.card(
                "Créneau le plus chargé",
                f"{DOW_LABELS[DOW_ORDER.index(int(peak['DOW']))]} {int(peak['PICKUP_HOUR'])}h",
                f"{nyc_theme.fmt_int(peak['TOTAL_TRIPS'])} courses cumulées")
        with c2:
            nyc_theme.card(
                "Créneau le plus creux",
                f"{DOW_LABELS[DOW_ORDER.index(int(low['DOW']))]} {int(low['PICKUP_HOUR'])}h",
                f"{nyc_theme.fmt_int(low['TOTAL_TRIPS'])} courses cumulées")
        with c3:
            nyc_theme.card("Rapport entre les deux",
                           f"×{peak['TOTAL_TRIPS'] / low['TOTAL_TRIPS']:.0f}",
                           "sur une même grille de 168 créneaux")

        # -- 1. Heatmap heure x jour --------------------------------------
        _title(
            "La semaine en 168 créneaux",
            "Une ligne par jour, une colonne par heure. Les profils « par jour » et "
            "« par heure » affichés séparément sont les deux marges de cette grille : "
            "chacun écrase ce que l'autre contient.",
        )
        fig = go.Figure(go.Heatmap(
            z=grid.values, x=[f"{h}h" for h in grid.columns], y=DOW_LABELS,
            colorscale=_heat(),
            hovertemplate="%{y} %{x}<br>%{z:,.0f} courses<extra></extra>",
            colorbar=dict(title=dict(text="courses", side="right"),
                          thickness=10, len=0.75),
            xgap=1, ygap=1,
        ))
        _fig(fig, 330, xaxis=dict(title="", tickmode="array",
                                  tickvals=[f"{h}h" for h in range(0, 24, 2)]),
             yaxis=dict(title="", autorange="reversed"))
        st.caption(
            "À 1 h du matin, le samedi porte sept fois plus de courses que le mardi : "
            "la nuit du vendredi se lit sur la ligne du samedi. À 8 h le rapport "
            "s'inverse, 445 000 courses le lundi contre 194 000 le dimanche. Un "
            "profil horaire moyen additionne les deux et n'en montre aucun."
        )

    # -- 2. Calendrier des 641 jours ---------------------------------------
    _title(
        "L'année, jour par jour",
        "Une case par jour, une colonne par semaine. Les creux ne sont pas du bruit : "
        "ce sont des dates.",
    )
    d = daily.sort_values("PICKUP_DATE").copy()
    first = d["PICKUP_DATE"].min()
    origin = first - pd.Timedelta(days=int(first.dayofweek))
    d["WEEK"] = ((d["PICKUP_DATE"] - origin).dt.days // 7).astype(int)
    d["DOW"] = d["PICKUP_DATE"].dt.dayofweek
    cal = d.pivot_table(index="DOW", columns="WEEK", values="TOTAL_TRIPS").reindex(range(7))

    # Un repère par mois, posé sur la semaine qui contient le premier du mois.
    starts = d[d["PICKUP_DATE"].dt.day == 1]
    ticks = starts["WEEK"].tolist()
    labels = [f"{dt:%b %y}" for dt in starts["PICKUP_DATE"]]

    fig = go.Figure(go.Heatmap(
        z=cal.values, x=cal.columns, y=DOW_LABELS,
        colorscale=_heat(),
        customdata=d.assign(TXT=d["PICKUP_DATE"].dt.strftime("%d %b %Y")).pivot_table(
            index="DOW", columns="WEEK", values="TXT", aggfunc="first").values,
        hovertemplate="%{customdata}<br>%{z:,.0f} courses<extra></extra>",
        colorbar=dict(title=dict(text="courses", side="right"),
                      thickness=10, len=0.75),
        xgap=2, ygap=2,
    ))
    worst = d.nsmallest(1, "TOTAL_TRIPS").iloc[0]
    best = d.nlargest(1, "TOTAL_TRIPS").iloc[0]
    for row, color, arrow in ((worst, p["WORSE"], "▼"), (best, p["BETTER"], "▲")):
        fig.add_annotation(
            x=row["WEEK"], y=DOW_LABELS[row["DOW"]],
            text=f"{arrow} {row['PICKUP_DATE']:%-d %b}", showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowcolor=color, ax=0, ay=-28,
            font=dict(size=10, color=color), bgcolor=p["BG_PAGE"],
            bordercolor=color, borderwidth=1, borderpad=2,
        )
    _fig(fig, 300,
         xaxis=dict(title="", tickmode="array", tickvals=ticks, ticktext=labels,
                    tickangle=-45, tickfont=dict(size=9)),
         yaxis=dict(title="", autorange="reversed"))

    # -- 3. Vitesse : semaine contre week-end ------------------------------
    if hw is not None and "AVG_SPEED_MPH" in hw.columns:
        _title(
            "La circulation ne suit pas le nombre de courses",
            "Le volume culmine le soir ; la ville est au plus lent en milieu "
            "d'après-midi. Deux lectures d'une même journée, séparées plutôt que "
            "posées sur deux axes.",
        )
        week = hw[hw["DOW"].isin([1, 2, 3, 4, 5])]
        wend = hw[hw["DOW"].isin([0, 6])]
        fig = go.Figure()
        for g, name, color, width in (
            (week, "Lundi – vendredi", p["ACCENT"], 2.4),
            (wend, "Samedi – dimanche", p["BAR_MUTED"], 2.0),
        ):
            # Moyenne pondérée à la main : la vitesse d'un créneau compte
            # pour le nombre de courses qu'il porte, pas pour une case.
            w = g.assign(NUM=g["AVG_SPEED_MPH"] * g["TOTAL_TRIPS"]).groupby("PICKUP_HOUR")
            s = (w["NUM"].sum() / w["TOTAL_TRIPS"].sum()).reindex(range(24))
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=s.tolist(), mode="lines", name=name,
                line=dict(color=color, width=width),
                hovertemplate="%{x}h<br>" + name + " : %{y:.1f} mph<extra></extra>",
            ))
        _fig(fig, 300,
             legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                         font=dict(size=10)),
             yaxis=dict(title="mph"),
             xaxis=dict(title="", tickmode="array", tickvals=list(range(0, 24, 2)),
                        ticktext=[f"{h}h" for h in range(0, 24, 2)]),
             margin=dict(t=28, b=10, l=0, r=0))
        st.caption(
            "Moyennes pondérées par le nombre de courses du créneau, distance divisée "
            "par la durée réelle, courses de 1 min à 3 h."
        )

    # -- 4. Évolution quotidienne ------------------------------------------
    _title(
        "Vingt-et-un mois de courses",
        "Le quotidien en trait fin, la moyenne sur sept jours par-dessus : la "
        "moyenne glissante retire l'oscillation hebdomadaire sans masquer les jours "
        "exceptionnels.",
    )
    d["MA7"] = d["TOTAL_TRIPS"].rolling(7, center=True, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["PICKUP_DATE"], y=d["TOTAL_TRIPS"], mode="lines", name="Jour",
        line=dict(color=p["BAR_MUTED"], width=0.9),
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} courses<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["PICKUP_DATE"], y=d["MA7"], mode="lines", name="Moyenne 7 jours",
        line=dict(color=p["ACCENT"], width=2.4),
        hovertemplate="%{x|%d %b %Y}<br>moy. 7 j : %{y:,.0f}<extra></extra>",
    ))
    _fig(fig, 320,
         legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                     font=dict(size=10)),
         yaxis=dict(title="courses par jour", rangemode="tozero"),
         xaxis=dict(title=""), margin=dict(t=28, b=10, l=0, r=0))


# ===========================================================================
# OÙ ? — d'où l'on part, où l'on arrive
# ===========================================================================
def where(zones, od, drop, airports):
    p = nyc_theme.palette()

    st.header("Où vont les courses")

    if od is None or drop is None:
        _missing("od_borough / dropoff_zones")
        return

    od = _known(od)
    total = od["TOTAL_TRIPS"].sum()
    same = od[od["PU_BOROUGH"] == od["DO_BOROUGH"]]
    cross = od[od["PU_BOROUGH"] != od["DO_BOROUGH"]]
    manh = same[same["PU_BOROUGH"] == "Manhattan"]

    c1, c2, c3 = st.columns(3)
    with c1:
        nyc_theme.card("Courses internes à un arrondissement",
                       f"{same['TOTAL_TRIPS'].sum() / total * 100:.0f} %",
                       f"dont {manh['TOTAL_TRIPS'].sum() / total * 100:.0f} % "
                       "dans Manhattan seul")
    with c2:
        nyc_theme.card("Courses qui changent d'arrondissement",
                       f"{cross['TOTAL_TRIPS'].sum() / total * 100:.0f} %",
                       f"${cross['TOTAL_REVENUE'].sum() / cross['TOTAL_TRIPS'].sum():.0f}"
                       f" par course, contre ${manh['TOTAL_REVENUE'].sum() / manh['TOTAL_TRIPS'].sum():.0f}"
                       " dans Manhattan")
    with c3:
        si = od[od["DO_BOROUGH"] == "Staten Island"]["TOTAL_TRIPS"].sum()
        nyc_theme.card("Arrivées à Staten Island",
                       nyc_theme.fmt_int(si),
                       f"soit {si / total * 100:.3f} % du total".replace(".", ",")
                       + " — le service y est résiduel")

    # -- 1. Sankey des flux entre arrondissements --------------------------
    _title(
        "Les 13 % de courses qui franchissent une frontière",
        "Les courses internes à un arrondissement sont retirées : à 87 % du total, "
        "elles écrasaient tout le reste en un seul ruban. Restent les liaisons, "
        "épaisseur proportionnelle au nombre de courses.",
    )
    f = cross[cross["TOTAL_TRIPS"] >= 5_000].copy()
    pu = [b for b in nyc_zones.BOROUGHS if b in set(f["PU_BOROUGH"])]
    do = [b for b in nyc_zones.BOROUGHS if b in set(f["DO_BOROUGH"])]
    nodes = [f"{b} (départ)" for b in pu] + [f"{b} (arrivée)" for b in do]
    idx_pu = {b: i for i, b in enumerate(pu)}
    idx_do = {b: len(pu) + i for i, b in enumerate(do)}

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=nodes, pad=18, thickness=14,
            color=[nyc_zones.color(b) for b in pu] + [nyc_zones.color(b) for b in do],
            line=dict(color=p["BG_PAGE"], width=0.5),
        ),
        link=dict(
            source=[idx_pu[b] for b in f["PU_BOROUGH"]],
            target=[idx_do[b] for b in f["DO_BOROUGH"]],
            value=f["TOTAL_TRIPS"].tolist(),
            color=[_rgba(nyc_zones.color(b), 0.34) for b in f["PU_BOROUGH"]],
            customdata=(f["TOTAL_REVENUE"] / f["TOTAL_TRIPS"]).round(2).tolist(),
            hovertemplate="%{source.label} → %{target.label}<br>"
                          "%{value:,.0f} courses<br>$%{customdata} par course<extra></extra>",
        ),
    ))
    _fig(fig, 400, font=dict(size=11, color=p["TEXT_LIGHT"]))
    st.caption(
        "Flux de moins de 5 000 courses écartés. Le premier d'entre eux, Queens → "
        "Manhattan, compte 3,7 millions de courses : c'est JFK et LaGuardia qui "
        "rentrent en ville."
    )

    # -- 2. Départs contre arrivées, par zone ------------------------------
    _title(
        "Les quartiers qui absorbent et ceux qui émettent",
        "Chaque point est une zone TLC. Sous la diagonale, on y monte plus qu'on "
        "n'en descend ; au-dessus, l'inverse. Les gares et les aéroports sont des "
        "portes à sens unique.",
    )
    z = zones.copy()
    z["ZONE_NAME"] = z["ZONE_ID"].map(nyc_zones.names())
    z["BOROUGH"] = z["ZONE_ID"].map(nyc_zones.boroughs())
    bal = z.merge(drop[["ZONE_ID", "TOTAL_TRIPS"]], on="ZONE_ID",
                  suffixes=("_PU", "_DO"))
    bal = bal.dropna(subset=["ZONE_NAME"])
    bal["RATIO"] = bal["TOTAL_TRIPS_DO"] / bal["TOTAL_TRIPS_PU"]

    # La diagonale part du plus petit point observé : tracée depuis une valeur
    # ronde, elle étirait l'échelle logarithmique sur deux décades vides.
    lo = min(bal["TOTAL_TRIPS_PU"].min(), bal["TOTAL_TRIPS_DO"].min()) * 0.8
    hi = max(bal["TOTAL_TRIPS_PU"].max(), bal["TOTAL_TRIPS_DO"].max()) * 1.25
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", hoverinfo="skip",
        line=dict(color=p["RULE"], width=1, dash="dot"), showlegend=False,
    ))
    for borough, g in bal.groupby("BOROUGH"):
        fig.add_trace(go.Scatter(
            x=g["TOTAL_TRIPS_PU"], y=g["TOTAL_TRIPS_DO"], mode="markers",
            name=borough, text=g["ZONE_NAME"],
            marker=dict(size=7, color=nyc_zones.color(borough), opacity=0.85,
                        line=dict(width=0.5, color=p["BG_PAGE"])),
            customdata=g["RATIO"],
            hovertemplate="<b>%{text}</b><br>%{x:,.0f} départs<br>"
                          "%{y:,.0f} arrivées<br>×%{customdata:.2f}<extra></extra>",
        ))
    # Les quatre zones les plus déséquilibrées dans chaque sens, nommées.
    named = pd.concat([bal.nlargest(4, "RATIO"), bal.nsmallest(4, "RATIO")])
    for i, row in enumerate(named.itertuples()):
        # Étiquettes alternées au-dessus et en dessous : les zones les plus
        # déséquilibrées se tiennent par paires dans le même coin du nuage.
        fig.add_annotation(
            x=np.log10(row.TOTAL_TRIPS_PU), y=np.log10(row.TOTAL_TRIPS_DO),
            text=row.ZONE_NAME, showarrow=False,
            yshift=12 if i % 2 == 0 else -12,
            font=dict(size=9, color=p["TEXT_DIM"]),
        )
    _fig(fig, 430,
         legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                     font=dict(size=10)),
         xaxis=dict(title="départs (échelle log)", type="log"),
         yaxis=dict(title="arrivées (échelle log)", type="log"),
         margin=dict(t=28, b=10, l=0, r=10))

    # -- 3. Les quinze premières destinations ------------------------------
    _title(
        "Les quinze premières destinations",
        "Barres classées plutôt qu'un treemap : l'œil compare des longueurs sur une "
        "base commune, pas des surfaces. La couleur porte l'arrondissement, pas une "
        "seconde mesure.",
    )
    top = drop.nlargest(15, "TOTAL_TRIPS").sort_values("TOTAL_TRIPS")
    fig = go.Figure(go.Bar(
        x=top["TOTAL_TRIPS"], y=top["ZONE_NAME"], orientation="h",
        marker=dict(color=[nyc_zones.color(b) for b in top["BOROUGH"]]),
        customdata=np.c_[top["AVG_FARE"], top["AVG_DISTANCE"]],
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} arrivées<br>"
                      "$%{customdata[0]:.2f} de tarif moyen<br>"
                      "%{customdata[1]:.1f} mi<extra></extra>",
    ))
    _fig(fig, 430, showlegend=False,
         xaxis=dict(title="courses à l'arrivée"),
         yaxis=dict(title=""), margin=dict(t=10, b=10, l=0, r=10))

    # -- 4. Aéroports mois par mois ----------------------------------------
    if airports is None:
        _missing("airport_monthly")
        return
    _title(
        "Les aéroports, dans les deux sens",
        "JFK et LaGuardia pèsent un dixième des courses. Les départs d'aéroport "
        "dominent largement les arrivées : on rentre en taxi, on part autrement.",
    )
    a = airports.copy()
    a["MONTH"] = pd.to_datetime(a["MONTH"])
    fig = go.Figure()
    styles = {
        ("JFK Airport", "Départ"): (p["ACCENT"], "solid"),
        ("JFK Airport", "Arrivée"): (p["ACCENT"], "dot"),
        ("LaGuardia Airport", "Départ"): (nyc_zones.color("Queens"), "solid"),
        ("LaGuardia Airport", "Arrivée"): (nyc_zones.color("Queens"), "dot"),
    }
    for (airport, direction), (color, dash) in styles.items():
        g = a[(a["AIRPORT"] == airport) & (a["DIRECTION"] == direction)]
        if g.empty:
            continue
        name = f"{airport.replace(' Airport', '')} — {direction.lower()}"
        fig.add_trace(go.Scatter(
            x=g["MONTH"], y=g["TOTAL_TRIPS"], mode="lines", name=name,
            line=dict(color=color, width=2, dash=dash),
            customdata=g["AVG_TOTAL"],
            hovertemplate="%{x|%b %Y}<br>" + name +
                          " : %{y:,.0f} courses<br>$%{customdata:.0f} en moyenne<extra></extra>",
        ))
    _fig(fig, 320,
         legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                     font=dict(size=10)),
         yaxis=dict(title="courses par mois", rangemode="tozero"),
         xaxis=dict(title=""), margin=dict(t=28, b=10, l=0, r=0))
    st.caption(
        "« Départ » compte les courses prises en charge à l'aéroport. Le tarif "
        "forfaitaire JFK – Manhattan ne dépend pas de la distance : il explique le "
        "palier visible dans la section Comment."
    )


# ===========================================================================
# COMMENT ? — la forme du trajet
# ===========================================================================
def how(profile, bins, payment, passengers):
    p = nyc_theme.palette()
    pr = profile.iloc[0]

    st.header("À quoi ressemble une course")

    # Médiane lue dans les bacs d'un demi-mile : la table `profile` ne porte
    # qu'une moyenne, et c'est précisément elle que la médiane corrige.
    median = None
    if bins is not None:
        cum = bins.sort_values("DISTANCE_BIN")["TOTAL_TRIPS"].cumsum()
        half = cum.iloc[-1] / 2
        median = bins.sort_values("DISTANCE_BIN")["DISTANCE_BIN"][cum >= half].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        nyc_theme.card(
            "Distance médiane",
            "—" if median is None else f"{median:.1f} mi".replace(".", ","),
            f"moyenne {_fr(pr['AVG_DISTANCE'])} mi, tirée par les aéroports")
    with c2:
        nyc_theme.card("Paiement par carte", f"{pr['PCT_CARTE']:.0f} %",
                       f"{100 - pr['PCT_CARTE']:.0f} % autrement")
    with c3:
        nyc_theme.card("Courses via un aéroport",
                       f"{pr['PCT_AEROPORT_JFK'] + pr['PCT_AEROPORT_LGA']:.0f} %",
                       "JFK et LaGuardia, départ ou arrivée")

    if bins is None:
        _missing("distance_bins")
    else:
        b = bins.copy()
        # Le dernier bac est un fourre-tout (tout ce qui dépasse 25 mi) :
        # le garder dans une distribution ferait une barre qui n'a pas la
        # même largeur que les autres.
        tail = b[b["DISTANCE_BIN"] >= 25]["TOTAL_TRIPS"].sum()
        b = b[b["DISTANCE_BIN"] < 25]

        # -- 1. Distribution des distances ---------------------------------
        _title(
            "Une ville de trajets courts, et un couloir vers l'aéroport",
            "Bacs d'un demi-mile. La masse est sous trois miles ; la bosse isolée "
            "vers 17 miles est le trajet Manhattan – JFK, que la moyenne de "
            f"{_fr(pr['AVG_DISTANCE'])} mi mélange avec le reste.",
        )
        colors = [
            p["ACCENT"] if 14 <= v <= 20 else p["BAR_MUTED"]
            for v in b["DISTANCE_BIN"]
        ]
        fig = go.Figure(go.Bar(
            x=b["DISTANCE_BIN"], y=b["TOTAL_TRIPS"], marker_color=colors,
            hovertemplate="%{x} – %{x} + 0,5 mi<br>%{y:,.0f} courses<extra></extra>",
        ))
        # Sur un axe logarithmique, une annotation se place en log10 de la
        # valeur : posée en valeur brute, elle part hors du cadre.
        fig.add_annotation(
            x=17,
            y=np.log10(b[b["DISTANCE_BIN"].between(14, 20)]["TOTAL_TRIPS"].max()),
            text="Manhattan – JFK", showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor=p["ACCENT"], ax=34, ay=-34,
            font=dict(size=10, color=p["ACCENT"]),
        )
        _fig(fig, 340, showlegend=False, bargap=0.08,
             xaxis=dict(title="distance en miles"),
             yaxis=dict(title="courses, échelle log", type="log"))
        st.caption(
            "Échelle verticale logarithmique : en linéaire, les 13 millions de "
            "courses du premier bac aplatissent tout le reste et la bosse aéroport "
            f"disparaît. {nyc_theme.fmt_int(tail)} courses de plus de 25 miles sont "
            "hors du graphique, pas du calcul : leur bac n'a pas la même largeur que "
            "les autres."
        )

        # -- 2. Tarif par tranche de distance ------------------------------
        _title(
            "Le prix cesse de suivre la distance à l'aéroport",
            "Montant moyen par bac de distance. La droite se brise vers 17 miles : "
            "le trajet Manhattan – JFK est au forfait, le compteur n'y court pas.",
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=b["DISTANCE_BIN"], y=b["AVG_TOTAL"], mode="lines",
            name="Montant total", line=dict(color=p["ACCENT"], width=2.4),
            hovertemplate="%{x} mi<br>$%{y:.2f} au total<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=b["DISTANCE_BIN"], y=b["AVG_FARE"], mode="lines",
            name="Compteur seul", line=dict(color=p["BAR_MUTED"], width=1.6),
            hovertemplate="%{x} mi<br>$%{y:.2f} au compteur<extra></extra>",
        ))
        fig.add_vrect(x0=15, x1=19, fillcolor=p["ACCENT"], opacity=0.07,
                      line_width=0, layer="below")
        _fig(fig, 320,
             legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                         font=dict(size=10)),
             xaxis=dict(title="distance en miles"),
             yaxis=dict(title="$ par course", rangemode="tozero"),
             margin=dict(t=28, b=10, l=0, r=0))

    # -- 3. Le pourboire mesure le terminal --------------------------------
    _title(
        "Le pourboire mesure le terminal, pas la générosité",
        "Un pourboire en espèces ne passe pas par le compteur : il n'existe pas dans "
        "la donnée. Tout taux calculé sur l'ensemble des courses est donc tiré vers "
        "le bas par la part payée en liquide.",
    )
    pay = payment.copy()
    pay["LABEL"] = pay["PAYMENT_TYPE"].map(PAYMENT_LABELS).fillna("Autre")
    pay["PART"] = pay["TOTAL_TRIPS"] / pay["TOTAL_TRIPS"].sum() * 100
    pay = pay[pay["PART"] >= 0.5].sort_values("PART")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pay["PART"], y=pay["LABEL"], orientation="h", name="Part des courses",
        marker_color=p["BAR_MUTED"],
        hovertemplate="<b>%{y}</b><br>%{x:.1f} % des courses<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=pay["PCT_AVEC_POURBOIRE"], y=pay["LABEL"], orientation="h",
        name="Courses avec pourboire", marker_color=p["ACCENT"],
        hovertemplate="<b>%{y}</b><br>%{x:.1f} % portent un pourboire<extra></extra>",
    ))
    _fig(fig, 260, barmode="group",
         legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                     font=dict(size=10)),
         xaxis=dict(title="", ticksuffix=" %", range=[0, 100]),
         yaxis=dict(title=""), margin=dict(t=28, b=10, l=0, r=10))

    # -- 4. Ce que la source déclare ---------------------------------------
    if passengers is None:
        _missing("passengers")
        return
    _title(
        "Ce que la source déclare, et ce qu'elle ne sait pas",
        "Le nombre de passagers est saisi par le chauffeur, pas mesuré. Les valeurs "
        "manquantes et les courses à zéro passager ne sont pas des erreurs de "
        "lecture : elles sont dans les fichiers de la TLC.",
    )
    pax = passengers.copy()
    pax["LABEL"] = pax["PASSENGERS"].apply(
        lambda v: "non renseigné" if v < 0 else f"{int(v)}"
    )
    pax["PART"] = pax["TOTAL_TRIPS"] / pax["TOTAL_TRIPS"].sum() * 100
    doubtful = pax[pax["PASSENGERS"] <= 0]["PART"].sum()

    fig = go.Figure(go.Bar(
        x=pax["LABEL"], y=pax["TOTAL_TRIPS"],
        marker_color=[
            p["WORSE"] if v <= 0 else p["BAR_MUTED"] for v in pax["PASSENGERS"]
        ],
        customdata=pax["PART"],
        hovertemplate="%{x} passager(s)<br>%{y:,.0f} courses"
                      "<br>%{customdata:.1f} % du total<extra></extra>",
    ))
    _fig(fig, 280, showlegend=False,
         xaxis=dict(title="passagers déclarés", type="category"),
         yaxis=dict(title="courses", type="log"))
    st.caption(
        f"Échelle verticale logarithmique : sans elle, les six dernières colonnes "
        f"seraient invisibles. {doubtful:.1f} % des courses déclarent zéro passager "
        f"ou rien du tout — de quoi écarter toute moyenne de passagers par course."
    )
