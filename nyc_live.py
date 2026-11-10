"""Les blocs animés du cockpit — ils vivent dans le navigateur, pas en Python.

Streamlit ne sait pas rejouer un cycle à l'écran : chaque image demanderait
un aller-retour serveur et une réexécution complète du script, soit deux ou
trois images par seconde avec un clignotement à chaque pas. Le temps
appartient donc au navigateur. Les données partent une fois, l'horloge tourne
en JavaScript, et Python n'est plus sollicité tant que le visiteur ne change
pas de page.

Quatre composants, quatre iframes de même origine :

    controls()  les boutons-panneaux, et l'horloge qui les accompagne
    city_map()  la carte à pics, hauteur des zones = mesure du pas
    profile()   le ruban qui défile, une barre par pas
    flows()     les flux entre arrondissements, à nœuds figés

Un seul composant tient l'horloge — `controls` — et publie l'état sur un
BroadcastChannel. Deux horloges indépendantes dériveraient l'une de l'autre
au bout de quelques secondes et la carte ne montrerait plus le pas affiché
sous le ruban.

L'état publié porte l'heure epoch du dernier changement de pas, pas une
position. Chaque composant en déduit sa propre phase entre deux pas et anime
en continu, à son rythme : le ruban glisse, les colonnes montent, sans que
personne n'ait à envoyer une image par trame.

Trois cycles, exclusifs :

    hour     24 pas, minuit -> minuit, tous jours confondus
    dow       7 pas, lundi -> dimanche, heures confondues
    month    12 pas, janvier -> décembre

Les valeurs sont des moyennes par heure du pas, calculées en amont par
scripts/J_build_cycles.py — un total brut ferait chuter novembre de moitié
pour la seule raison qu'il n'est observé qu'une fois sur la période, et
gonflerait chaque pas de l'année d'un facteur vingt-quatre parce qu'il porte
un mois entier. L'unité est donc la même sur les trois cycles, et un bloc de
la carte y vaut partout vingt courses à l'heure.

Transport : le binaire et les métadonnées sont servis depuis `static/`.
Streamlit renvoie le HTML des composants à chaque réexécution du script, et
des données embarquées repasseraient par la connexion à chaque clic.
"""

import json
import pathlib
import shutil
import uuid

import numpy as np
import pandas as pd
import streamlit as st

import nyc_theme
import nyc_zones

ROOT = pathlib.Path(__file__).resolve().parent
STATIC = ROOT / "static"
GEO = ROOT / "data" / "reference" / "taxi_zones.json"
# La maquette de la ville : 2 601 emprises de bâtiments avec leur hauteur de
# toit, tirées une fois du cadastre ouvert de New York. Aucun script du dépôt
# ne la reconstruit — c'est un export unique, pas un calcul — donc elle est
# versionnée comme les contours de zones et copiée telle quelle vers static/.
# Sans cette copie, le conteneur de production n'a pas de tours : le fichier
# n'existait que dans static/, qui est hors du dépôt.
CITY = ROOT / "data" / "reference" / "nyc_city.json"
ZONE_CYCLE = ROOT / "data" / "aggregates" / "zone_cycle.parquet"
OD_CYCLE = ROOT / "data" / "aggregates" / "od_cycle.parquet"
DUR_CYCLE = ROOT / "data" / "aggregates" / "dur_cycle.parquet"

CHANNEL = "nyc-cockpit"


def _channel() -> str:
    """Nom du canal, propre à l'onglet.

    Un BroadcastChannel porte au-delà du document : deux onglets ouverts sur
    l'application se retrouvaient avec deux horloges maîtresses sur le même
    nom, chacune avançant le pas de l'autre. On voyait le ruban et la carte
    sauter d'un pas, et le cycle changer tout seul.

    La clé vit dans st.session_state, qui est propre à la session du
    navigateur : les cinq composants d'une même page partagent donc leur
    canal, et deux pages ne se parlent pas.
    """
    key = "_cockpit_channel"
    if key not in st.session_state:
        st.session_state[key] = uuid.uuid4().hex[:8]
    return f"{CHANNEL}-{st.session_state[key]}"

# Deux mesures, pas trois. La distance moyenne a été retirée du sélecteur :
# elle ne variait que de x1,41 sur la journée, contre x10,7 pour les courses
# et x9,9 pour les revenus, et un cockpit dont le principe est que tout
# bouge au même pas accueille mal une mesure qui ne bouge pas. Elle était
# aussi la seule des trois à n'être pas une quantité mais une moyenne, ce
# qui lui valait un titre à part, une repondération dans le ruban et une
# unité de bloc à elle.
#
# Elle n'est pas perdue pour autant : le tableau de bord la porte toujours,
# au totalisateur du bloc des nombres, où une moyenne se lit bien. Ce qui
# disparaît est son panneau et son tiers de binaire — 92 ko.
MEASURES = ["trips", "revenue"]
COLUMNS = {"trips": "TOTAL_TRIPS", "revenue": "TOTAL_REVENUE"}

DOW_LABELS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi",
              "Samedi", "Dimanche"]
MONTH_LABELS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                "Juillet", "Août", "Septembre", "Octobre", "Novembre",
                "Décembre"]

# Le ruban écrit un libellé sous chaque barre, à quelques dizaines de pixels
# d'intervalle : « Mercredi » et « Septembre » s'y chevauchent. Les commandes,
# elles, gardent le nom entier — elles ont la place.
DOW_SHORT = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
MONTH_SHORT = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
               "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

# Durée d'un pas, en millisecondes.
#
# La semaine a porté l'heure dans son pas un temps — 168 pas, un par heure de
# chaque journée — pour garder le cycle jour/nuit visible d'un lundi à un
# dimanche. Les trois boucles duraient alors 40,8 s, donc la semaine avançait
# par pas de 243 ms, et à ce rythme les instruments ne se lisaient plus : une
# aiguille qui change de place quatre fois par seconde est une vibration, pas
# une mesure. La semaine revient à sept pas, un par jour.
#
# Ce qui est perdu est dit ici : un pas de « mardi » moyenne ses vingt-quatre
# heures, donc le cycle jour/nuit n'apparaît plus dans ce cycle. Il reste
# entier dans le cycle de la journée, qui est fait pour lui.
#
# Le pas de la semaine est celui de la journée, 1 700 ms : c'est le rythme
# déjà éprouvé du cycle le plus regardé, et sept pas de 5 829 ms — ce qu'une
# boucle de 40,8 s aurait demandé — tiennent chaque jour à l'écran six
# secondes, ce qui n'avance plus. La boucle de la semaine fait donc 11,9 s
# contre 40,8 s pour les deux autres.
CYCLES = {
    "hour": {"steps": 24, "ms": 1700, "label": "Heure",
             "labels": [f"{h}h" for h in range(24)],
             "short": [f"{h}h" for h in range(24)]},
    "dow": {"steps": 7, "ms": 1700, "label": "Jour",
            "labels": DOW_LABELS, "short": DOW_SHORT},
    "month": {"steps": 12, "ms": 3400, "label": "Mois",
              "labels": MONTH_LABELS, "short": MONTH_SHORT},
}

# Zones trop rares pour porter une colonne : moins d'une course par jour en
# moyenne sur la période. Les garder allongerait le binaire sans rien ajouter.
MIN_TRIPS_PER_DAY = 1.0


def available() -> bool:
    return (GEO.exists() and ZONE_CYCLE.exists() and OD_CYCLE.exists()
            and DUR_CYCLE.exists())


# Numéro de forme du payload. Il entre dans la clé du cache : sans lui, un
# changement de structure ne serait pas vu tant que les agrégats ne sont pas
# reconstruits, et le composant lirait un fichier à l'ancien format.
PAYLOAD_VERSION = 17


@st.cache_data(show_spinner=False)
def _build(version: int, geo_mtime: float, zone_mtime: float,
           od_mtime: float, dur_mtime: float) -> dict:
    """Écrit les fichiers statiques et rend un résumé.

    Les dates de modification sont des arguments et non des lectures
    internes : elles font la clé du cache, qui s'invalide donc tout seul
    quand les agrégats sont reconstruits.

    Le cache de Streamlit ne vit qu'en mémoire : chaque redémarrage du
    conteneur refaisait ce calcul pour le premier visiteur. Un tampon posé à
    côté des fichiers garde la même clé sur disque ; s'il correspond, les
    fichiers déjà écrits sont repris tels quels.
    """
    # dur_mtime fait partie de la clé : il manquait, et reconstruire la seule
    # table des durées laissait donc le tampon valide et les fichiers périmés.
    key = {"version": version,
           "mtimes": [geo_mtime, zone_mtime, od_mtime, dur_mtime]}
    stamp = STATIC / "nyc_live.stamp.json"
    if stamp.exists() and (STATIC / "nyc_live.bin").exists() \
            and (STATIC / "nyc_live.json").exists():
        saved = json.loads(stamp.read_text())
        if saved.get("key") == key:
            return saved["summary"]
    summary = _compute()
    stamp.write_text(json.dumps({"key": key, "summary": summary}))
    return summary


def _compute() -> dict:
    geo = {z["id"]: z for z in json.loads(GEO.read_text())}
    zc = pd.read_parquet(ZONE_CYCLE)
    od = pd.read_parquet(OD_CYCLE)
    dur = pd.read_parquet(DUR_CYCLE)

    # Le volume d'une journée moyenne sert de filtre : c'est le cycle
    # horaire qui le porte, puisque ses 24 pas couvrent une journée entière.
    daily = zc[zc["SCOPE"] == "hour"].groupby("ZONE_ID")["TOTAL_TRIPS"].sum()
    kept = [int(z) for z in daily[daily >= MIN_TRIPS_PER_DAY].index if z in geo]

    blobs, scales = [], {}
    for scope, cycle in CYCLES.items():
        part = zc[(zc["SCOPE"] == scope) & (zc["ZONE_ID"].isin(kept))]
        scales[scope] = {}
        for key in MEASURES:
            grid = (part.pivot_table(index="ZONE_ID", columns="STEP",
                                     values=COLUMNS[key], aggfunc="sum")
                        .reindex(index=kept, columns=range(cycle["steps"]))
                        .fillna(0.0)
                        .to_numpy())
            packed, scale = _pack(grid)
            blobs.append(packed.tobytes())
            scales[scope][key] = scale

    STATIC.mkdir(exist_ok=True)
    (STATIC / "nyc_live.bin").write_bytes(b"".join(blobs))

    meta = {
        "zones": [
            {"id": z, "name": geo[z]["name"], "borough": geo[z]["borough"],
             "lon": geo[z]["lon"], "lat": geo[z]["lat"],
             "polygon": _outer_ring(geo[z]["geometry"])}
            for z in kept
        ],
        "measures": MEASURES,
        "cycles": {k: {**v, "scales": scales[k]} for k, v in CYCLES.items()},
        "city": _city(zc, od),
        "dur": _dur(dur),
        "flows": _flows(od),
        "areas": nyc_zones.AREAS,
        "areaColors": nyc_zones.AREA_COLORS,
    }
    (STATIC / "nyc_live.json").write_text(json.dumps(meta, separators=(",", ":")))
    return {"zones": len(kept),
            "bin_ko": (STATIC / "nyc_live.bin").stat().st_size // 1024,
            "json_ko": (STATIC / "nyc_live.json").stat().st_size // 1024}


# Colonnes de zone_cycle.parquet dont le bloc des nombres a besoin, et que
# scripts/J_build_cycles.py écrit depuis la version 5 du payload.
CITY_COLUMNS = ["TOTAL_TRIPS", "TOTAL_REVENUE", "TRIPS_PAX", "PAX_SOLO",
                "PAX_DUO", "PAX_TRIO", "TRIPS_PAY", "PAY_TIP", "PAY_CARD",
                "PAY_CASH", "SUM_MINUTES", "MILES"]

# Nombre de bacs de dur_cycle.parquet, qui doit valoir DUR_BINS du script J.
# Il sert à réindexer : un pas dont un bac est vide n'a pas de ligne dans le
# Parquet, et la case attend un tableau de longueur fixe.
DUR_BINS = 31


def _air(od: pd.DataFrame, scope: str, steps: int) -> dict:
    """Les deux parts des courses d'aéroport, en pourcentage du pas.

    Mesurées sur od_cycle et non sur zone_cycle : celui-ci ne connaît que la
    zone de départ, et c'est justement la dissymétrie des deux sens qui fait
    l'intérêt de la case. Les départs vers un aéroport se massent à 4 h et
    5 h — 13,6 % des courses à 5 h contre 0,05 % à minuit — tandis que les
    retours d'aéroport se répartissent sur la journée avec deux bosses,
    16 h et 23 h. Les deux parts culminent donc aux antipodes de l'horloge.

    Les quelques courses d'un aéroport à un autre sont comptées au départ :
    elles existent, elles doivent tomber d'un côté, et elles partent bien
    d'un aéroport. Les deux parts et leur complément font ainsi 100 % sans
    reste.

    Aucune course n'est exclue : contrairement au nombre de passagers et au
    moyen de paiement, la zone de départ et celle d'arrivée sont renseignées
    partout. Cette case n'a donc pas de base à déclarer.
    """
    air = nyc_zones.AIRPORT_AREA
    part = od[od["SCOPE"] == scope]
    pu, do = part["PU_AREA"] == air, part["DO_AREA"] == air
    g = part.assign(
        FROM=part["TOTAL_TRIPS"].where(pu, 0.0),
        TO=part["TOTAL_TRIPS"].where(do & ~pu, 0.0),
    ).groupby("STEP")[["TOTAL_TRIPS", "FROM", "TO"]].sum()
    g = g.reindex(range(steps)).fillna(0.0)
    tot = g["TOTAL_TRIPS"].replace(0.0, np.nan)
    return {"fromair": g["FROM"] / tot * 100, "toair": g["TO"] / tot * 100}


def _city(zc: pd.DataFrame, od: pd.DataFrame) -> dict:
    """Les séries de la ville entière, un tableau par pas et par mesure.

    Le binaire porte une grille par zone et par mesure : c'est ce qu'il faut
    à la carte, qui extrude chaque zone. Le bloc des nombres, lui, ne lit
    jamais une zone — il somme la ville. Faire voyager six grilles de plus
    pour n'en lire que les totaux doublerait le binaire ; ces séries-là font
    quelques centaines de nombres et tiennent en clair dans le JSON.

    Les rapports sont faits ici et non dans le navigateur parce qu'ils
    pèsent sur les numérateurs : une part se calcule sur la somme des
    courses, jamais sur la moyenne des parts de zones — sans quoi une zone
    de trois courses compterait autant que Midtown.
    """
    missing = [c for c in CITY_COLUMNS if c != "MILES" and c not in zc.columns]
    if missing:
        raise RuntimeError(
            "zone_cycle.parquet est d'une version antérieure : "
            + ", ".join(missing) + " manquent. "
            "Relancer scripts/J_build_cycles.py."
        )
    out = {}
    for scope, cycle in CYCLES.items():
        part = zc[zc["SCOPE"] == scope]
        # Les milles parcourus, zone par zone, avant de sommer : AVG_DISTANCE
        # est une moyenne, et une somme de moyennes ne veut rien dire.
        part = part.assign(MILES=part["AVG_DISTANCE"] * part["TOTAL_TRIPS"])
        g = (part.groupby("STEP")[CITY_COLUMNS].sum()
                 .reindex(range(cycle["steps"])).fillna(0.0))
        pax = g["TRIPS_PAX"].replace(0.0, np.nan)
        pay = g["TRIPS_PAY"].replace(0.0, np.nan)
        trips = g["TOTAL_TRIPS"].replace(0.0, np.nan)
        hours = (g["SUM_MINUTES"] / 60.0).replace(0.0, np.nan)
        miles = g["MILES"].replace(0.0, np.nan)
        # Deux dénominateurs, et non TOTAL_TRIPS : les courses sans
        # passenger_count et celles sans moyen de paiement déclaré sont les
        # mêmes 15,66 % de lignes, et leur taux suit l'heure — 9,9 % à 16 h,
        # 39,6 % à 4 h. Au dénominateur, elles feraient raconter leurs trous
        # aux deux cases plutôt que les passagers et les paiements.
        series = {
            "solo": g["PAX_SOLO"] / pax * 100,
            "duo": g["PAX_DUO"] / pax * 100,
            "trio": g["PAX_TRIO"] / pax * 100,
            "tip": g["PAY_TIP"] / pay * 100,
            "card": g["PAY_CARD"] / pay * 100,
            "cash": g["PAY_CASH"] / pay * 100,
            "mph": g["MILES"] / hours,
            "dur": g["SUM_MINUTES"] / trips,
            "dist": g["MILES"] / trips,
            # La part des courses sur laquelle la répartition des passagers
            # est calculée. Elle est affichée sous la case, et elle bouge
            # autant que ce qu'elle qualifie : 60 % des courses déclarent un
            # passager à 4 h, 90 % à 16 h. Sans elle, « 85,7 % seul » ne dit
            # pas sur combien de courses il porte.
            #
            # Les trois parts de paiement et leur base restent calculées
            # bien qu'aucune case ne les lise depuis la version 8 : elles
            # font quelques centaines d'octets, et c'est le seul endroit du
            # projet où l'on sait que payment_type = 0 et passenger_count
            # vide sont exactement les mêmes 15,66 % de lignes.
            "paxbase": g["TRIPS_PAX"] / trips * 100,
            "paybase": g["TRIPS_PAY"] / trips * 100,
            # Le prix du mile : les dollars encaissés divisés par les milles
            # parcourus, et non la moyenne des prix du mile de chaque zone.
            # Il raconte l'embouteillage, parce que le compteur facture au
            # temps dès que le taxi ralentit — 5,83 $ le mile à 5 h quand la
            # ville est vide, 9,37 $ à 18 h quand elle est bloquée. Son
            # corollaire est qu'il suit la vitesse de très près, r = -0,94
            # sur l'heure : les deux vont donc dans la même case, en miroir,
            # au lieu d'en occuper deux à dire la même chose.
            "mile": g["TOTAL_REVENUE"] / miles,
        }
        series.update(_air(od, scope, cycle["steps"]))
        out[scope] = {k: [round(float(x), 4) for x in v.fillna(0.0)]
                      for k, v in series.items()}
    return out


def _dur(dur: pd.DataFrame) -> dict:
    """La distribution des durées, un tableau de parts par pas.

    En pourcentage des courses du pas et non en comptes : la case compare la
    forme d'un pas à la forme moyenne du cycle, et deux pas n'ont pas le même
    volume — 739 courses à 4 h contre 7 939 à 18 h. En comptes, la case ne
    montrerait que l'affluence, qui est déjà tout l'histogramme.

    Les bacs vides n'ont pas de ligne dans le Parquet : d'où la réindexation
    sur les 31 bacs, sans laquelle un tableau court décalerait la courbe.
    """
    out = {}
    for scope, cycle in CYCLES.items():
        part = dur[dur["SCOPE"] == scope]
        grid = (part.pivot_table(index="STEP", columns="BIN",
                                 values="TOTAL_TRIPS", aggfunc="sum")
                    .reindex(index=range(cycle["steps"]),
                             columns=range(DUR_BINS))
                    .fillna(0.0)
                    # En flottants : les comptes sortent du Parquet en
                    # entiers, et la division en place refuserait d'y écrire
                    # un rapport.
                    .to_numpy(dtype=float))
        total = grid.sum(axis=1, keepdims=True)
        share = np.divide(grid * 100.0, total, out=np.zeros_like(grid),
                          where=total > 0)
        out[scope] = [[round(float(x), 3) for x in row] for row in share]
    return out


def _pack(grid: np.ndarray) -> tuple:
    """Grille de flottants vers entiers 16 bits, et son échelle.

    L'échelle est prise sur le maximum observé : la mesure occupe toute
    l'amplitude des 16 bits et aucune valeur n'est écrêtée. Un flottant
    pèserait quatre fois plus pour une précision dont une colonne extrudée
    n'a aucun usage.
    """
    top = float(np.nanmax(grid)) if grid.size else 0.0
    scale = top / 65_535 if top > 0 else 1.0
    return np.rint(np.nan_to_num(grid) / scale).astype("<u2"), scale


def _flows(od: pd.DataFrame) -> dict:
    """Flux entre secteurs, en clair : quelques milliers de nombres.

    Assez petit pour voyager en JSON. Les flux internes à un secteur sont
    gardés ici — c'est le composant qui décidera de les montrer ou non, et
    ils servent au total de chaque nœud.
    """
    out = {}
    valid = set(nyc_zones.AREAS)
    od = od[od["PU_AREA"].isin(valid) & od["DO_AREA"].isin(valid)]
    for scope, cycle in CYCLES.items():
        part = od[od["SCOPE"] == scope]
        rows = []
        for (pu, do), g in part.groupby(["PU_AREA", "DO_AREA"]):
            series = (g.set_index("STEP")["TOTAL_TRIPS"]
                       .reindex(range(cycle["steps"]))
                       .fillna(0.0))
            if series.sum() < 1:
                continue
            rows.append({"s": pu, "t": do,
                         "v": [round(float(x), 1) for x in series]})
        out[scope] = rows
    return out


def _outer_ring(geometry: dict) -> list:
    """L'anneau extérieur le plus étendu d'une géométrie.

    Plusieurs zones sont des MultiPolygon — Staten Island et les Rockaways
    traînent des îlots. Une colonne extrudée se pose sur une surface : on
    garde la plus grande et on laisse les îlots, puisqu'une zone ne porte
    qu'une valeur de toute façon.
    """
    if geometry["type"] == "Polygon":
        rings = [geometry["coordinates"][0]]
    else:
        rings = [poly[0] for poly in geometry["coordinates"]]
    return max(rings, key=len)


def meta() -> dict:
    _serve_city()
    return _build(PAYLOAD_VERSION, GEO.stat().st_mtime,
                  ZONE_CYCLE.stat().st_mtime, OD_CYCLE.stat().st_mtime,
                  DUR_CYCLE.stat().st_mtime)


def _serve_city() -> None:
    """Pose la maquette de ville dans static/, si elle n'y est pas déjà.

    Hors du cache de _build : la copie ne dépend d'aucun agrégat, et laissée
    dedans elle n'avait lieu qu'au premier calcul. Un processus au cache
    chaud dont static/ vient d'être recréé se retrouvait alors sans tours.
    """
    if not CITY.exists():
        return
    out = STATIC / "nyc_city.json"
    if out.exists() and out.stat().st_size == CITY.stat().st_size:
        return
    STATIC.mkdir(exist_ok=True)
    shutil.copyfile(CITY, out)


# ---------------------------------------------------------------------------
# Rendu des composants
# ---------------------------------------------------------------------------
def _cfg(**extra) -> str:
    p = nyc_theme.palette()
    return json.dumps({
        "channel": _channel(),
        "mode": nyc_theme.mode(),
        "colors": {
            "bg": p["BG_PAGE"], "ink": p["TEXT_LIGHT"], "dim": p["TEXT_DIM"],
            "rule": p["RULE"], "accent": p["ACCENT"], "panel": p["BG_PANEL"],
            "brick": p["BRICK"],
            "muted": p["BAR_MUTED"],
        },
        **extra,
    })


def _html(body: str, cfg: str) -> str:
    return _SHELL.replace("__CFG__", cfg).replace("__BODY__", body)


def _head(title: str, hint: str = "", **extra) -> dict:
    return {"title": title, "hint": hint, **extra}


def controls(height: int = 160, scope: str = "hour",
             measure: str = "trips") -> None:
    """Les boutons-panneaux et l'horloge du cockpit."""
    meta()
    st.iframe(
        _html(_controls_body(), _cfg(scope=scope, measure=measure)),
        height=height,
    )


def kpis(height: int = 160) -> None:
    """Quatre nombres qui suivent le pas de l'horloge."""
    meta()
    st.iframe(_html(_KPIS, _cfg()), height=height)


# Angle de vue de la carte. Le centre et l'échelle, eux, ne sont pas fixés
# ici : le composant les calcule sur l'emprise réelle des zones et sur la
# taille du bloc. Une valeur en dur se décrochait dès que la grille changeait
# de proportions — la ville sortait du cadre ou flottait au milieu du noir.
VIEW = {"pitch": 50, "bearing": -18}


def city_map(height: int = 460, title: str = "", hint: str = "") -> None:
    """La carte à pics. Hauteur et couleur suivent la mesure du pas."""
    meta()
    # Le titre se pose sur le sol de la carte, pas sur la page : il prend donc
    # l'encre de ce sol — claire sur la ville de nuit, sombre sur celle de jour.
    head = (_head(title, hint, ink="#f2efe6", dim="rgba(242,239,230,0.62)")
            if nyc_theme.mode() == "dark" else
            _head(title, hint, ink="#1b1b1f", dim="rgba(27,27,31,0.58)"))
    st.iframe(_html(_MAP, _cfg(view=VIEW, head=head)), height=height)


def taxi_city(height: int = 460, title: str = "",
              hint: str = "", legend: bool = False) -> None:
    """La maquette de New York, un taxi par course.

    Les tours posent le sol et ne portent aucune donnée : elles situent. Ce
    qui porte la donnée est le jaune, et lui seul — d'où une carte en gris
    violacé où rien d'autre n'est chaud.
    """
    meta()
    head = (_head(title, hint, ink="#f2efe6", dim="rgba(242,239,230,0.62)")
            if nyc_theme.mode() == "dark" else
            _head(title, hint, ink="#1b1b1f", dim="rgba(27,27,31,0.58)"))
    st.iframe(_html(_CITY, _cfg(head=head, legend=legend)), height=height)


def profile(height: int = 240, title: str = "", hint: str = "") -> None:
    """Le ruban de barres qui défile vers la gauche."""
    meta()
    st.iframe(_html(_PROFILE, _cfg(head=_head(title, hint))), height=height)


def flows(height: int = 300, title: str = "", hint: str = "",
          legend: bool = False) -> None:
    """Les flux entre arrondissements, nœuds figés, rubans animés."""
    meta()
    st.iframe(
        _html(_FLOWS, _cfg(head=_head(title, hint), legend=legend)),
        height=height)


# ---------------------------------------------------------------------------
# Socle commun à tous les composants : chargement du payload, canal, horloge.
# ---------------------------------------------------------------------------
_SHELL = r"""
<!doctype html>
<meta charset="utf-8">
<style>
  html, body { margin: 0; height: 100%; overflow: hidden;
               font-family: "Source Sans Pro", system-ui, sans-serif;
               -webkit-font-smoothing: antialiased; }
  #load { position: absolute; inset: 0; display: flex; align-items: center;
          justify-content: center; font-size: 0.8rem; }
  #tip { position: absolute; pointer-events: none; padding: 7px 10px;
         border-radius: 4px; font-size: 0.76rem; line-height: 1.45;
         opacity: 0; transition: opacity 0.12s; white-space: nowrap;
         z-index: 9; }
  /* Le titre de chaque bloc vit dans son iframe et non dans la page : les
     hauteurs d'iframe sont fixes, et un titre posé par Streamlit au-dessus
     ajouterait une marge différente selon le bloc, donc deux colonnes qui ne
     s'alignent plus. Posé en surimpression, il ne prend aucune place dans la
     mise en page et laisse le dessin occuper toute la surface. */
  #head { position: absolute; left: 0; top: 0; right: 0; z-index: 8;
          pointer-events: none; line-height: 1.3; }
  #head b { display: block; font-size: 0.88rem; font-weight: 600; }
  #head i { display: block; font-size: 0.72rem; font-style: normal; }
  /* La légende vit au bas de l'iframe, en surimpression, pour la même
     raison que le titre. Les deux blocs du bas portent la même : ce sont
     les mêmes secteurs dans les mêmes couleurs, et c'est elle qui fait que
     la carte et le ruban se lisent ensemble. */
  #legend { position: absolute; left: 0; right: 0; bottom: 0; z-index: 8;
            pointer-events: none; display: flex; flex-wrap: wrap;
            gap: 2px 11px; font-size: 0.66rem; line-height: 1.3; }
  #legend span { display: inline-flex; align-items: center; gap: 5px;
                 white-space: nowrap; }
  #legend i { width: 8px; height: 8px; border-radius: 2px;
              flex: 0 0 auto; }
</style>
<div id="load">…</div>
<div id="head"></div>
<div id="legend"></div>
<div id="tip"></div>
<script>
const CFG = __CFG__;
const C = CFG.colors;
document.body.style.background = C.bg;
document.body.style.color = C.ink;
document.getElementById("load").style.color = C.dim;

/* Hauteur réservée au titre, que chaque composant retire de sa surface utile.
   Mesurée et non fixée : la phrase d'aide se replie sur deux ou trois lignes
   selon la largeur du bloc, et une valeur en dur faisait passer le dessin
   sous le texte dès que le bloc rétrécissait. */
let HEAD = 0;
const HEAD_EL = document.getElementById("head");
let headMeasure = () => {};
(function () {
  if (!CFG.head || !CFG.head.title) { HEAD_EL.remove(); return; }
  HEAD_EL.innerHTML = "<b>" + CFG.head.title + "</b>"
              + (CFG.head.hint ? "<i>" + CFG.head.hint + "</i>" : "");
  HEAD_EL.querySelector("b").style.color = CFG.head.ink || C.ink;
  const hint = HEAD_EL.querySelector("i");
  if (hint) { hint.style.color = CFG.head.dim || C.dim; }
  headMeasure = () => { HEAD = HEAD_EL.offsetHeight + 8; };
  headMeasure();
  window.addEventListener("resize", headMeasure);
})();

/* Réécrit le titre d'un bloc. Un bloc dont le dessin change de mesure ou de
   cycle change aussi de sujet : un titre figé y devient faux, et le ruban
   annonçait « le volume » même quand on avait allumé les dollars.

   La hauteur est reprise après coup : un titre plus long se replie sur deux
   lignes, et le dessin doit alors descendre d'autant. */
function headTitle(text) {
  const b = HEAD_EL.querySelector && HEAD_EL.querySelector("b");
  if (!b || b.textContent === text) { return; }
  b.textContent = text;
  headMeasure();
}

/* Ce que porte l'axe des ordonnées, mesure par mesure, et rien de plus. Le
   titre ne dit pas quel cycle tourne : « une journée type » nommait une
   lecture du dessin, pas ce qui y est mesuré, et le cycle se lit déjà sur
   le sélecteur juste en dessous et sur les étiquettes de l'axe.

   « par heure » est vrai sur les trois cycles, et c'est un acquis du
   builder, pas une approximation : J_build_cycles.py divise par le nombre
   d'occurrences du pas fois le nombre d'heures qu'il couvre, donc un pas de
   mois est ramené à l'heure comme un pas d'heure. La ville fait 743 à 7 981
   courses/h sur la journée et 4 151 à 5 305 sur l'année : deux amplitudes
   différentes, une seule unité.

   Les deux mesures sont des quantités horaires, donc elles partagent la
   même phrase d'unité. C'est ce qui reste après le retrait de la distance,
   qui était une moyenne et cassait la règle. */
const MEASURE_TITLE = { trips: "Volume de courses par heure",
                        revenue: "Revenus par heure" };

/* Pendant de HEAD, en bas. La légende se replie sur une, deux ou trois
   lignes selon la largeur du bloc, donc sa hauteur se mesure. Pas `FOOT` :
   la maquette de ville nomme déjà ainsi l'empreinte au sol de ses tours, et
   deux déclarations du même nom au premier niveau du script faisaient
   tomber le composant entier. */
let LEGH = 0;
const LEG_EL = document.getElementById("legend");
/* Le secteur des aéroports en porte trois, et il n'en vaut que deux. Cumulé
   sur les vingt-quatre pas du cycle de la journée : JFK 5 119 courses à
   l'heure, LaGuardia 3 382, Newark 3,3 — soit 60,2 %, 39,8 % et 0,04 % du
   secteur. Les taxis jaunes ne chargent pas à Newark, qui est hors de l'État
   de New York : trois courses sur 8 504.

   La légende nomme donc les deux qui comptent. « Airports » laissait croire
   à un secteur à trois portes, alors que la carte n'en dessine que deux —
   et qu'elle a raison de n'en dessiner que deux. Le nom du secteur lui-même
   ne change pas : il est écrit dans les agrégats, où il est une clé. */
const AREA_SHOW = { Airports: "Airports · JFK + LGA" };

function drawLegend() {
  if (!CFG.legend || !META.areas) { LEG_EL.remove(); return; }
  LEG_EL.innerHTML = META.areas.map(n =>
    '<span><i style="background:' + (META.areaColors[n] || C.dim)
    + '"></i>' + (AREA_SHOW[n] || n) + "</span>").join("");
  LEG_EL.style.color = (CFG.head && CFG.head.dim) || C.dim;
  const measure = () => { LEGH = LEG_EL.offsetHeight + 8; };
  measure();
  window.addEventListener("resize", measure);
}

const TIP = document.getElementById("tip");
TIP.style.background = C.panel;
TIP.style.border = "1px solid " + C.rule;
TIP.style.color = C.ink;

function tipAt(x, y, html) {
  if (!html) { TIP.style.opacity = 0; return; }
  TIP.innerHTML = html;
  TIP.style.left = Math.min(x + 14, window.innerWidth - TIP.offsetWidth - 8) + "px";
  TIP.style.top = Math.min(y + 14, window.innerHeight - TIP.offsetHeight - 8) + "px";
  TIP.style.opacity = 1;
}

const FR = n => Math.round(n).toLocaleString("fr-FR");
function fmt(v, measure) {
  if (measure === "revenue") { return "$" + FR(v); }
  return FR(v) + " courses";
}
const MEASURE_LABEL = { trips: "Courses", revenue: "Revenus" };

/* Ordre d'affichage des trois mesures, découplé de META.measures — qui est
   l'ordre physique du binaire et ne peut pas bouger sans le réécrire.
   Courses passe en dernier : c'est la mesure par défaut, mais le ruban juste
   au-dessus la montre déjà barre par barre, donc elle n'a pas besoin de la
   place où l'œil arrive d'abord. */
const MEASURE_ORDER = ["revenue", "trips"];
function measuresInOrder() {
  const known = MEASURE_ORDER.filter(k => META.measures.indexOf(k) >= 0);
  return known.concat(META.measures.filter(k => MEASURE_ORDER.indexOf(k) < 0));
}

/* L'iframe est servie en srcdoc : son URL de base ne mène nulle part et une
   requête relative échouerait. La page hôte est de même origine, on y lit
   l'adresse des fichiers statiques. */
function staticBase() {
  try {
    const l = window.parent.location;
    return l.origin + l.pathname.replace(/\/$/, "") + "/app/static/";
  } catch (e) { return "/app/static/"; }
}

let META = null, BIN = null, Z = 0, READY = false;
/* Table des séries : SERIES[cycle][mesure] est un Uint16Array de Z x pas,
   rangé zone par zone. L'ordre de lecture suit celui de l'écriture côté
   Python — cycles dans l'ordre du dictionnaire, mesures dans l'ordre de la
   liste. */
const SERIES = {};

/* État partagé. `at` est l'heure epoch du dernier changement de pas : chaque
   composant en déduit sa phase et anime en continu sans recevoir une image
   par trame. Date.now() et non performance.now(), dont l'origine diffère
   d'une iframe à l'autre. */
/* La boucle démarre arrêtée si le système demande moins de mouvement.
   WCAG 2.2.2 « Pause, Stop, Hide » est déjà tenu par le bouton des
   commandes, et le bouton reste la réponse exigée : la préférence ne le
   remplace pas. Mais l'inverse est vrai aussi. Les trois cycles tournent
   40,8 s en boucle sans fin, bien au-delà des cinq secondes du critère ;
   quelqu'un qui a posé la préférence au niveau du système ne devrait pas
   avoir à désarmer cinq iframes à la main à chaque visite. Il voit donc le
   premier pas, à l'arrêt, et part de là.

   Chaque iframe fait la lecture pour son compte et toutes tombent d'accord,
   la préférence étant la même pour toute la page. */
const CALM = typeof matchMedia === "function"
  && matchMedia("(prefers-reduced-motion: reduce)").matches;

let S = { scope: "hour", measure: "trips", step: 0, playing: !CALM,
          at: Date.now() };

const channel = new BroadcastChannel(CFG.channel);

function phase() {
  if (!S.playing) { return 0; }
  const ms = META.cycles[S.scope].ms;
  return Math.max(0, Math.min(1, (Date.now() - S.at) / ms));
}

function stepsOf(scope) { return META.cycles[scope].steps; }

function value(z, step, scope, measure) {
  const n = stepsOf(scope);
  return SERIES[scope][measure][z * n + ((step % n) + n) % n];
}

function scaled(z, step, scope, measure) {
  return value(z, step, scope, measure) * META.cycles[scope].scales[measure];
}

async function load() {
  const base = staticBase();
  META = await (await fetch(base + "nyc_live.json")).json();
  BIN = await (await fetch(base + "nyc_live.bin")).arrayBuffer();
  Z = META.zones.length;
  let offset = 0;
  for (const scope of Object.keys(META.cycles)) {
    SERIES[scope] = {};
    const n = META.cycles[scope].steps;
    /* Plus de maximum ni de minimum relevés ici : ils ne servaient qu'à
       l'échelle relative de la carte, que les blocs ont remplacée. C'était
       un balayage de toutes les séries à chaque chargement de chacune des
       cinq iframes — 332 640 entiers, cinq fois, pour rien. */
    for (const m of META.measures) {
      SERIES[scope][m] = new Uint16Array(BIN, offset, Z * n);
      offset += Z * n * 2;
    }
  }
  drawLegend();
  READY = true;
  const el = document.getElementById("load");
  if (el) { el.remove(); }
}

channel.onmessage = e => {
  const m = e.data;
  if (!m || m.from === "state") { return; }
  if (m.type === "state") {
    S = m.state;
    /* Le payload peut n'être pas encore là : le premier dessin aura lieu
       dans boot(), avec le dernier état reçu. La garde porte sur READY et
       non sur META, qui est posé avant que les séries ne soient découpées —
       un état reçu entre les deux faisait lire un tableau inexistant. */
    if (READY && typeof onState === "function") { onState(); }
  }
};

/* `load` et non `DOMContentLoaded` : le corps du composant peut tirer une
   bibliothèque externe, et boot() ne doit pas partir avant qu'elle soit là. */
const ready = new Promise(r => {
  if (document.readyState === "complete") { r(); }
  else { window.addEventListener("load", r); }
});

Promise.all([load(), ready]).then(() => {
  if (typeof boot === "function") { boot(); }
  /* Un composant qui arrive après les autres ne connaît pas l'état courant :
     il le demande, le maître de l'horloge le republie. */
  channel.postMessage({ type: "hello" });
}).catch(err => {
  /* La console d'abord : une fois le voile de chargement retiré, un échec
     de boot() ne laisserait aucune trace visible. */
  console.error("composant", err);
  const el = document.getElementById("load");
  if (el) { el.textContent = "Indisponible : " + err.message; }
});
</script>
__BODY__
"""

# ---------------------------------------------------------------------------
# Composant 1 — les boutons-panneaux, et l'horloge
# ---------------------------------------------------------------------------
def _controls_body() -> str:
    """Les boutons du cockpit, en signalisation de voie rapide.

    Ce sont eux qui disent où l'on va dans les données : la mesure à gauche,
    le pas du cycle à droite. De jour ils prennent la signalisation qu'on lit
    en roulant sur le Brooklyn-Queens Expressway — une armature d'acier, la
    destination en cours sur tôle verte, les autres sur plaque blanche, et la
    flèche de voie qui n'apparaît qu'au survol. Un panneau blanc indique ; il
    n'oriente que lorsqu'on le regarde.

    Chaque groupe pend de sa propre travée, et il n'y a pas de montants : un
    arceau fermé, pas des poteaux plantés sur le trottoir. L'armature est un
    treillis — deux membrures et une course de croix de Saint-André — parce
    qu'une barre lisse fait du plastique, et elle passe derrière le bord haut
    des plaques, visible aux deux bouts du groupe.

    Elle est vue en léger plongeon vers son milieu : la ferme du fond est la
    même ferme réduite autour du centre de la travée, et quatre entretoises
    joignent les deux plans aux quatre coins. Sans elles, deux fermes ne font
    que deux rectangles emboîtés.

    Elle est dessinée en SVG mesuré par trussSvg(), à la largeur réelle du
    groupe, parce qu'un nombre entier de panneaux est la seule façon de finir
    la course sur un sommet. Forme, puis matière, puis ombres : la silhouette
    est posée une fois dans un masque et tout le reste est peint à travers
    elle, en rectangles pleins. Dessinée pièce par pièce, chaque recouvrement
    se voyait.

    Le pas courant est du mobilier lui aussi, mais pas un panneau : ce n'est
    pas une destination. C'est une borne, posée entre les voies, au bas du
    rail — un fût à sommet arrondi, bande de couleur en tête, chiffre en
    noir. Les plaques pendent du portique, la borne tient au sol.

    Les six plaques ont une seule taille, le mot centré dessus : une travée
    se commande en série, et six largeurs au gabarit du texte faisaient six
    objets là où il n'y en a qu'un, répété.

    La flèche est celle d'une affectation de voies, posée sous le mot et non
    à côté : droit devant sur la plaque en cours, à quarante-cinq degrés sur
    les autres. Le MUTCD n'en connaît pas d'autres angles, et les deux
    suffisent à dire « on y va » et « on y irait ».

    De nuit, pas de portique : un panneau rétroréfléchissant sans phare pour
    l'éclairer n'est qu'une plaque grise. La ligne garde alors le cerclage
    chromé et le faisceau ambré du phare, et le clignotant au survol.
    """
    m = nyc_theme.materials()
    dark = m["mode"] == "dark"
    p = nyc_theme.palette()
    accent = p["ACCENT"]

    gantry = "" if dark else """
  /* L'armature. Une travée en caisson par groupe de plaques, vue de face et
     à peu près à la hauteur des panneaux : ils se boulonnent sur sa face
     avant et la couvrent, sauf aux deux bouts du groupe.

     Elle est dessinée par trussSvg(), en SVG mesuré, et non plus en dégradés
     répétés. Un `repeating-linear-gradient` est un champ infini qu'on rogne à
     la largeur du conteneur : la course de X s'arrêtait donc où elle tombait,
     au milieu d'une barre. Pour finir sur un sommet il faut connaître la
     largeur et y poser un nombre entier de panneaux.

     Le dégagement à chaque bout vaut un panneau entier — sous cette valeur la
     diagonale passe derrière la plaque avant d'avoir fini sa montée, et le X
     se lit coupé. */
  /* Les deux groupes prennent la même part du rail, et leurs plaques sont
     centrées dedans. Ils comptaient deux plaques à gauche et trois à droite,
     donc deux travées d'inégale portée : celle de gauche n'avait que cinq
     panneaux de croix là où l'autre en avait sept, et le même objet se
     lisait deux fois à deux échelles. Sur une voie rapide les travées d'un
     même échangeur ont la même portée, et c'est le nombre de panneaux
     suspendus qui change, pas la poutre. */
  /* Une grille à trois colonnes et non `space-between` : les deux colonnes
     en `1fr` sont égales, et leur plancher est le contenu le plus large des
     deux. En flex, répartir l'espace restant à parts égales gardait l'écart
     des contenus — deux plaques à gauche, trois à droite — et les travées
     n'avaient pas la même portée : cinq panneaux de croix d'un côté, sept de
     l'autre, le même objet lu à deux échelles. Sur une voie rapide les
     travées d'un même échangeur ont la même portée, et c'est le nombre de
     plaques suspendues qui change, pas la poutre. */
  #rail { display: grid; grid-template-columns: 1fr auto 1fr;
         align-items: center; }
  .set { padding: 0 38px; min-width: 0; justify-content: center; }
  .truss { position: absolute; left: 0; right: 0; top: 4px; height: 46px;
          z-index: 0; opacity: 0.95; }
  .beam { position: absolute; inset: 0; display: block; }
  /* Deux ombres, une par plan, parce qu'une ombre est dure au contact et
     molle avec la distance : la largeur de la pénombre suit le rapport entre
     la distance de l'objet à la surface et sa distance à la lumière. La ferme
     du fond est près du panneau blanc, celle de devant en est loin. La même
     ombre pour les deux était le défaut.

     `drop-shadow` et non `box-shadow` : le premier suit le contour réellement
     dessiné — les diagonales et les vides entre elles — quand le second
     n'ombre que la boîte de l'élément, un rectangle plein. */
  .beam.far { filter:
      drop-shadow(0.5px 1px 0.5px hsl(var(--sh) 22% / 0.46))
      drop-shadow(1px 2px 1.5px hsl(var(--sh) 26% / 0.26)); }
  .beam.near { filter:
      drop-shadow(2px 4px 4px hsl(var(--sh) 30% / 0.26))
      drop-shadow(4px 8px 10px hsl(var(--sh) 36% / 0.15)); }
  /* La matière de l'acier galvanisé, masquée au métal seul. Étalée sur la
     bande entière, elle remplissait les vides d'un voile gris et le croisé
     disparaissait dessous. */
  .spangle { mix-blend-mode: overlay; opacity: 0.55; }
  .draw { mix-blend-mode: multiply; opacity: 0.231; }
  .soil { mix-blend-mode: multiply; opacity: 0.303; }
  /* Une travée se commande en une seule taille : les plaques sont
     identiques, le mot centré dessus. */
  .sign { box-sizing: border-box; flex: 0 0 auto; width: 104px;
          text-align: center; padding: 11px 6px 26px; }
"""

    # De jour la plaque est un caisson d'aluminium extrudé : tôle peinte,
    # bord blanc intérieur à la norme, et aucune fixation visible — la face
    # d'un panneau de direction est lisse.
    #
    # Elle portait aussi deux joncs chromés de deux pixels en haut et en bas,
    # reste de la plaque de rue sertie dans ses rails. Ils occupaient les
    # mêmes arêtes que l'arête roulée du caisson, donc chaque bord avait deux
    # contours. Un objet à la fois.
    WHITE_FACE = ("linear-gradient(152deg, #ffffff 0%, #f6f6f4 46%,"
                  " #ececeb 100%)")

    def plate(face):
        if dark:
            return ""
        return f"background-image: {face};"

    on_face = plate(m["sign_face"])
    on_ink = accent if dark else m["sign_ink"]
    off_face = plate(WHITE_FACE)
    off_ink = p["TEXT_DIM"] if dark else "#15151a"

    # Les arêtes du caisson et l'ombre portée, sous une seule lumière.
    #
    # L'ombre portée n'avait aucun décalage horizontal alors que la lumière
    # vient de la gauche, et elle était grise sur un fond chaud. Une ombre
    # tombe du côté opposé à sa source, son décalage vertical vaut le double
    # de l'horizontal, et elle prend la teinte de ce sur quoi elle tombe,
    # assombrie — le gris est le défaut qui trahit le calque. Les couches
    # doublent maintenant à chaque fois, le flou suivant le décalage
    # vertical, et la première est l'ombre de contact : le trait presque sans
    # flou là où la plaque touche la poutre. C'est son absence qui fait
    # flotter un objet.
    #
    # Sur les arêtes, le terme de Fresnel : une surface vue en incidence
    # rasante réfléchit plus. Le bas et la droite du caisson ont donc un
    # liseré clair malgré l'ombre, juste à l'intérieur du trait sombre. Un
    # bord uniformément noir est le même défaut que l'ombre grise.
    #
    # Le reflet est légèrement froid et l'ombre franchement chaude : c'est ce
    # que fait une lumière de ciel sur un sol clair.
    #
    # Rayon de courbure : la norme le prend à un douzième du petit côté, soit
    # quatre pixels et demi ici.
    caisson = "" if dark else """
  :root { --sh: 32deg 14%; }
  .sign { --sign-r: 5px; border: 0; border-radius: var(--sign-r);
          box-shadow:
            inset 0 1px 0 hsl(210deg 26% 98% / 0.52),
            inset 1px 0 0 hsl(210deg 22% 97% / 0.24),
            inset 0 -1px 0 hsl(var(--sh) 16% / 0.22),
            inset -1px 0 0 hsl(var(--sh) 16% / 0.14),
            inset 0 -2px 0 hsl(42deg 24% 86% / 0.20),
            inset -2px 0 0 hsl(42deg 24% 86% / 0.12),
            0.5px 1px 1px hsl(var(--sh) 26% / 0.44),
            1px 2px 2px hsl(var(--sh) 30% / 0.26),
            2px 4px 4px hsl(var(--sh) 34% / 0.18),
            4px 8px 8px hsl(var(--sh) 38% / 0.11); }
"""

    # Le pas courant n'est pas une destination : il n'a donc pas de plaque.
    # Il est posé entre les voies, à l'endroit où l'on trouve une borne — un
    # fût trapu à sommet arrondi, bande de couleur en tête, chiffre en noir.
    # Et une borne tient au sol : elle descend au bas du rail quand les
    # panneaux pendent du portique.
    STONE = ("linear-gradient(100deg, #fbfaf7 0%, #efede8 52%,"
             " #e2dfd8 100%)")
    clock = ("" if dark else f"""
  #clock {{ align-self: flex-end; align-items: flex-end; gap: 8px; }}
  #now {{ position: relative; z-index: 1; box-sizing: border-box;
         width: 52px; min-width: 0; padding: 13px 0 5px;
         text-align: center; font-size: 0.9rem; font-weight: 800;
         text-transform: uppercase; letter-spacing: 0.02em; color: #15151a;
         border-radius: 10px 10px 2px 2px;
         background-image: linear-gradient(to bottom,
             {p['BRICK']} 0%, color-mix(in srgb, {p['BRICK']} 82%, #000) 100%),
             {STONE};
         background-size: 100% 9px, 100% 100%;
         background-position: top left, center;
         background-repeat: no-repeat; }}
  #play {{ position: relative; z-index: 1; width: 24px; flex: 0 0 24px;
          padding: 3px 0; font-size: 0.64rem; border-radius: 2px;
          background: {STONE}; color: #2a2a31;
          border: 1px solid rgba(21, 21, 26, 0.22); }}
""")

    # La flèche d'affectation de voie a quitté cette feuille : elle était un
    # masque CSS, donc une image tramée à dix-neuf pixels qu'on faisait
    # tourner. À 90° elle tombait sur la grille des pixels et restait nette,
    # à 45° elle s'empâtait — deux flèches qui ne se ressemblaient pas. C'est
    # maintenant un <path> tourné dans le SVG, en vectoriel, dans la racine
    # fantôme du composant. Voir laneSvg() et laneAngle().

    # Les deux matières de nuit — cerclage chromé et clignotant — n'existent
    # qu'en thème sombre, et il ne suffit pas de vider leur `content` en
    # thème clair : ces règles visent les mêmes pseudo-éléments que la
    # flèche, et une déclaration plus spécifique l'effaçait au survol.
    # De nuit, la barre est une voiture vue de dessus. À gauche ce qu'on
    # mesure : les phares, lumière blanc chaud qui ouvre la route. À droite le
    # pas du cycle : les feux arrière, rouge brique, qui ne projettent rien.
    # Un phare et un feu rouge ne se confondent jamais, et c'est ce qui
    # distingue les deux groupes sans leur donner deux géométries.
    #
    # L'optique reste celle de la barre latérale : cerclage chromé contre le
    # mot, flaque qui décroît vers la gauche, poussières qui dérivent dans le
    # faisceau. Le clignotant, lui, devient un feu de recul — blanc et fixe.
    # Un clignotant annonce qu'on va tourner ; ici on regarde une destination
    # qu'on n'a pas prise.
    dust = m["dust"]
    # Un feu arrière est franchement rouge : c'est une lampe derrière un
    # verre teinté, pas un mur de brique.
    tail = "#e23b27"
    # Un phare n'est pas ambré : il est blanc à peine chaud. L'ambre, lui,
    # reste la couleur des feux de la ville, pas celle de la lumière projetée.
    front = "#fff2d4"
    # Le mot n'est pas rouge : il est éclairé par du rouge. C'est donc un
    # blanc qui a pris la teinte de la lampe, pas la lampe elle-même — un
    # libellé à la saturation du verre se lisait comme une alerte.
    tail_ink = "#f6b2a0"
    reverse = "#f4f1ea"

    def pool(colour, near, mid, mid_at, far, to="left"):
        """Flaque de lumière : elle décroît du cerclage vers le mot.

        La fin est écrite `color-mix(... 0%, transparent)` et non `transparent`
        tout court : le mot-clé vaut `rgba(0,0,0,0)`, et interpoler vers lui
        traverse du noir, ce qui laissait un trait au bout du dégradé. La
        couleur de fin est donc la même teinte, à opacité nulle.

        Elle s'éteint aussi avant le bord : une flaque qui touche l'arête du
        bouton y dessine une ligne nette, et une lumière n'a pas d'arête.
        """
        out = f"color-mix(in srgb, {colour} 0%, transparent)"
        return (f"linear-gradient(to {to},"
                f" color-mix(in srgb, {colour} {near}%, transparent) 0%,"
                f" color-mix(in srgb, {colour} {mid}%, transparent) {mid_at}%,"
                f" {out} {far}%, {out} 100%)")

    def lamp(colour, strong, soft):
        """Halo d'une source ponctuelle : un proche dense, un large diffus."""
        return (f"0 0 5px 0 color-mix(in srgb, {colour} {strong}%, transparent),"
                f" 0 0 16px 3px color-mix(in srgb, {colour} {soft}%, transparent)")

    # Le mot est éclairé par sa propre lampe : blanc chaud sous le phare,
    # rouge sous le feu arrière. Un libellé ambré au-dessus d'un verre rouge
    # disait deux lumières là où il n'y en a qu'une.
    lamp_ink = "" if not dark else (
        f"#measures .sign.on {{ color: {front}; }}"
        f" #scopes .sign.on {{ color: {tail_ink}; }}")

    night = "" if not dark else f"""
  /* Le texte se tient contre le métal : la place du cerclage est réservée
     sur tous les boutons, sinon ils se décalent en s'allumant. */
  /* Pas de bordure de nuit : le fond est peint jusque sous elle, et ce
     dernier pixel se lisait comme un filet le long du bouton. */
  .sign {{ padding-right: 17px; background-color: transparent;
          border: 0; padding-left: 13px; border-radius: 0;
          background-clip: padding-box; }}
  #measures .sign.on {{ background-image: {pool(front, 26, 9, 44, 92)}; }}
  /* Pleins phares : survoler la ligne déjà active monte l'intensité. */
  #measures .sign.on:hover {{ background-image: {pool(front, 42, 18, 48, 92)}; }}
  /* Un feu arrière éclaire ses abords, il n'ouvre pas la route : sa flaque
     s'arrête à 42 % de la largeur quand celle du phare va au bout. */
  /* Le feu arrière éclaire vers l'arrière : la voiture regarde à gauche, la
     lumière part donc vers la droite. Sa lampe est côté intérieur, comme
     celle des phares, et les deux groupes éclairent vers l'extérieur. */
  #scopes .sign {{ padding-left: 17px; padding-right: 13px; }}
  #scopes .sign.on {{ background-image: {pool(tail, 58, 20, 16, 44, 'right')}; }}
  #scopes .sign.on:hover {{ background-image: {pool(tail, 80, 30, 18, 48, 'right')}; }}
  /* Les poussières dérivent dans le faisceau, sinon ce sont des taches. Elles
     vivent sur leur propre couche pour que la lumière, elle, ne bouge pas.
     Rien derrière un feu arrière : il n'y a pas de faisceau où voir voler
     quoi que ce soit. */
  /* La poussière est découpée par son propre dégradé et non par un
     `overflow: hidden` sur le bouton : un coin arrondi qui rogne un fond
     laisse une couture d'un pixel le long de l'arête. */
  #measures .sign.on::before {{
      content: ""; position: absolute; inset: 0 3px 0 0; pointer-events: none;
      background-image: {dust['image']};
      background-size: {dust['size']};
      background-repeat: {dust['repeat']};
      background-position: {dust['start']};
      animation: nyc-dust 7s linear infinite alternate; }}
  @keyframes nyc-dust {{
      from {{ background-position: {dust['start']}; }}
      to   {{ background-position: {dust['end']}; }} }}
  /* Le cerclage est une pièce et non un trait : une épaisseur, des bouts
     arrondis, un retrait haut et bas qui le détache du bord. Devant c'est du
     chrome ; derrière, le verre rouge du feu, qui s'éclaire lui-même. */
  .sign.on::after {{
      content: ""; position: absolute;
      top: 1px; bottom: 1px; right: 0; width: 3px; border-radius: 2px; }}
  #measures .sign.on::after {{
      background: {m['metal']};
      box-shadow: inset 0 0 0 0.5px rgba(255,255,255,0.08); }}
  #scopes .sign.on::after {{
      left: 0; right: auto;
      background: {tail}; box-shadow: {lamp(tail, 85, 40)}; }}
  /* À l'avant, le clignotant : survoler une destination où l'on ne va pas
     encore, c'est signaler son intention d'y aller. Ambré, battant à 1,5 Hz —
     la cadence d'un vrai feu, entre 60 et 120 battements par minute. */
  #measures .sign:not(.on):hover::before {{
      background-image: {pool(accent, 38, 15, 14, 28)};
      animation: nyc-blink 0.68s linear infinite; }}
  #measures .sign:not(.on):hover::after {{
      background: {accent}; box-shadow: {lamp(accent, 85, 45)};
      animation: nyc-blink 0.68s linear infinite; }}
  @keyframes nyc-blink {{
      0%, 42% {{ opacity: 1; }}
      48%, 100% {{ opacity: 0; }} }}
  @media (prefers-reduced-motion: reduce) {{
      #measures .sign:not(.on):hover::before,
      #measures .sign:not(.on):hover::after {{
          animation: none; opacity: 0.55; }} }}
  /* À l'arrière, le feu de recul : blanc, fixe, et court comme tout feu
     arrière. Un clignotant annonce qu'on tourne ; un feu de recul dit
     seulement qu'on regarde derrière. */
  .sign:not(.on):hover::before {{
      content: ""; position: absolute;
      left: 0; right: 0; top: 50%; height: 18px; margin-top: -9px;
      pointer-events: none;
      background-image: {pool(reverse, 26, 10, 14, 30)}; }}
  #scopes .sign:not(.on):hover::before {{
      background-image: {pool(reverse, 26, 10, 14, 30, 'right')}; }}
  #scopes .sign:not(.on):hover::after {{ left: 0; right: auto; }}
  .sign:not(.on):hover::after {{
      content: ""; position: absolute; top: 50%; right: 0;
      width: 4px; height: 14px; margin-top: -7px;
      border-radius: 2px; background: {reverse}; pointer-events: none;
      box-shadow: {lamp(reverse, 50, 20)}; }}
  @media (prefers-reduced-motion: reduce) {{
      #measures .sign.on::before {{ animation: none; }} }}
"""

    return (
        rf"""
<style>
  /* Une seule ligne, trois groupes écartés aux extrémités : ce qu'on mesure
     à gauche, l'horloge au milieu, le pas du cycle à droite. L'écart suffit
     à les distinguer — ni intitulé « Cycle », ni intitulé « Mesure », qui ne
     faisaient que nommer ce que les mots disaient déjà.

     Le rail porte le portique : c'est lui qui donne au treillis sa largeur
     et aux montants leurs deux bouts. Les plaques pendent de son bord haut,
     donc elles s'alignent par le haut et non par le milieu — des panneaux
     d'inégale profondeur sur une même poutre, pas trois boutons centrés. */
  #wrap {{ position: absolute; inset: 0; display: flex; align-items: center;
          padding: 0 2px; }}
  #rail {{ position: relative; flex: 1 1 auto; display: flex;
          align-items: center; justify-content: space-between; }}
  #clock {{ display: flex; align-items: center; gap: 10px; }}
  .set {{ position: relative; z-index: 1; display: flex;
         align-items: flex-start; gap: 6px; }}
  .sign {{ position: relative; z-index: 1; border: 1px solid transparent;
          background: none; cursor: pointer; font: inherit;
          font-size: 0.72rem; font-weight: {'600' if dark else '800'};
          text-transform: uppercase; letter-spacing: 0.07em;
          border-radius: 2px;
          padding: {'6px 12px' if dark else '8px 14px 22px'};
          color: {off_ink};
          transition: color 0.18s ease;
          {off_face} }}
  .sign.on {{ color: {on_ink};
             {"font-weight: 700; letter-spacing: 0.11em;" if dark else ""}
             {on_face} }}
  {lamp_ink}
  .sign.on:hover {{ color: {on_ink}; }}
  .sign:not(.on):hover {{ color: {p['TEXT_LIGHT'] if dark else '#15151a'}; }}
  #play {{ font: inherit; font-size: 0.72rem; cursor: pointer; width: 30px;
          padding: 5px 0; border-radius: 3px; flex: 0 0 30px;
          background: {p['BG_PANEL']}; color: {p['TEXT_LIGHT']};
          border: 1px solid {p['RULE']}; }}
  #now {{ font-size: 1.15rem; font-weight: 600;
         font-variant-numeric: tabular-nums; min-width: 72px;
         text-align: right; color: {p['TEXT_LIGHT']}; }}
  {gantry}
  {caisson}
  {clock}
  {night}
</style>
<div id="wrap">
  <div id="rail">
    <div id="measures" class="set"></div>
    <div id="clock">
      <div id="now"></div>
      <button id="play"></button>
    </div>
    <div id="scopes" class="set"></div>
  </div>
</div>
"""
        + _CONTROLS_JS
    )

_CONTROLS_JS = r"""
<script>
/* La police des plaques. Overpass est une interprétation libre des alphabets
   FHWA Series — le « Highway Gothic » des panneaux routiers américains. Elle
   est servie depuis static/fonts/, pas depuis un CDN : le conteneur n'a pas
   forcément de réseau sortant, et une plaque qui retombe en police système
   perd ce qui la fait reconnaître comme plaque. Licence SIL OFL 1.1, texte
   dans static/fonts/OFL-Overpass.txt.

   Chargée par l'API FontFace et non par une règle @font-face : l'adresse des
   fichiers statiques n'est connue qu'à l'exécution — l'iframe est servie en
   srcdoc et n'a pas d'URL de base utilisable. Les fontes ajoutées au registre
   du document valent aussi dans les racines fantômes. */
(function loadPlateFont() {
  if (!window.FontFace || !document.fonts) { return; }
  const base = staticBase() + "fonts/";
  /* Deux sous-ensembles : le latin de base, et l'étendu pour les accents des
     trois langues de l'interface. Le navigateur ne télécharge l'étendu que
     s'il rencontre un caractère qui s'y trouve. */
  const subsets = [
    ["overpass-latin.woff2", "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC,"
      + " U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F,"
      + " U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD"],
    ["overpass-latin-ext.woff2", "U+0100-02BA, U+02BD-02C5, U+02C7-02CC,"
      + " U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, U+1D00-1DBF,"
      + " U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0,"
      + " U+2113, U+2C60-2C7F, U+A720-A7FF"],
  ];
  for (const [file, range] of subsets) {
    try {
      const f = new FontFace("Overpass", "url(" + base + file + ")",
                             { weight: "400 800", unicodeRange: range,
                               display: "swap" });
      f.load().then(x => document.fonts.add(x)).catch(() => {});
    } catch (e) { /* une plaque en police système reste lisible */ }
  }
})();

/* <nyc-sign> — une plaque de rue.

   Le panneau était un <button> dont toute la matière tenait dans des
   dégradés CSS. Un dégradé est lisse, et une plaque de rue ne l'est pas :
   elle est couverte d'un film de microbilles de verre qui a du grain, et ses
   arêtes s'usent. C'est ce lissé qui la faisait reconnaître comme un dessin
   et non comme un objet.

   L'élément garde la tôle et les joncs sur son hôte — la feuille de style du
   composant les pose comme avant, et c'est Python qui tient la palette. Il
   ajoute dans sa racine fantôme deux calques de bruit réel, produits par
   feTurbulence, et le mot dans la police des panneaux.

   Pourquoi une racine fantôme : les deux calques sont du décor et ne doivent
   jamais être atteints par une règle de la page, ni atteindre quoi que ce
   soit. Et chaque plaque y porte ses propres identifiants de filtre, donc sa
   propre graine de bruit — six plaques au grain identique se liraient comme
   six copies d'une image.

   L'hôte n'est plus un <button>, il en reprend donc le contrat : rôle,
   tabulation, Entrée et Espace, et aria-pressed qui dit l'état. */
const SIGN_CSS = `
  :host { display: inline-block; position: relative; }
  /* L'encre est sérigraphiée sur le film, pas posée dessus : elle prend donc
     la même lumière. Le liseré est le ras de lettre — l'épaisseur d'encre
     accroche la lumière sur son bord haut. Sur le vert, les lettres blanches
     halent un peu : un film rétroréfléchissant diffuse sous le tracé, et
     c'est à ça qu'on reconnaît une plaque éclairée d'une plaque imprimée. */
  .t { position: relative; z-index: 3; display: block;
       font-family: Overpass, "Helvetica Neue", Arial, sans-serif;
       text-shadow: 0 0.5px 0 rgba(255,255,255,0.55),
                    0 -0.5px 0.5px rgba(0,0,0,0.22); }
  :host(.on) .t { text-shadow: 0 0 2px rgba(238,255,246,0.40),
                               0 0.5px 0 rgba(0,0,0,0.30); }
  /* La peau rogne tout le décor au rayon de la plaque. Sans elle, les SVG de
     matière peignent un carré plein : un hôte ne rogne pas ses enfants, et le
     grain comme l'usure débordaient aux quatre coins par-dessus l'arrondi. */
  .skin { position: absolute; inset: 0; z-index: 1; overflow: hidden;
          border-radius: inherit; pointer-events: none; }
  /* Le grain du film passe par-dessus l'encre : il est à la surface, et
     l'encre est dessous. Des lettres nettes sur une tôle grenue, c'était un
     calque, pas une impression. */
  .skin.over { z-index: 4; }
  .ink { mix-blend-mode: overlay;
         opacity: calc(var(--sign-grain, 0.15) * 0.45); }
  /* Son rayon se déduit de celui de la plaque moins son retrait, sinon
     l'écart entre l'arête du caisson et le bord blanc n'est plus constant
     dans les angles. */
  .edge { position: absolute; z-index: 2; pointer-events: none;
          inset: calc(var(--sign-edge-w, 1.5px) * 2.6);
          border: var(--sign-edge-w, 1.5px) solid var(--edge, #fff);
          border-radius: max(0px, calc(var(--sign-r, 5px)
                       - var(--sign-edge-w, 1.5px) * 2.6)); }
  /* La rétroréflexion : le film renvoie la lumière vers sa source, donc la
     tôle a un foyer hors axe et non un éclairement uniforme. Le foyer est
     étiré en largeur — un film prismatique a une direction de billes, et un
     halo rond se lit comme une bulle posée dessus.

     Deux sources, pas une : le ciel en haut à gauche, et le rebond de la
     chaussée en bas à droite, quatre fois plus faible. C'est ce second terme,
     presque invisible, qui empêche le bas de tomber dans un noir plat. */
  .sheen { position: absolute; z-index: 1; inset: 0; pointer-events: none;
           mix-blend-mode: screen; opacity: var(--sign-sheen, 0.35);
           background:
             radial-gradient(150% 108% at 26% -26%,
               rgba(255,255,255,0.78) 0%, rgba(255,255,255,0.24) 38%,
               rgba(255,255,255,0.00) 68%),
             radial-gradient(120% 70% at 78% 128%,
               rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.00) 62%); }
  /* L'ombre propre : le bas d'un panneau est plus sombre que le haut, parce
     qu'il ne voit qu'une moitié de ciel. */
  .ambient { position: absolute; z-index: 1; inset: 0; pointer-events: none;
             background: linear-gradient(to bottom,
               rgba(0,0,0,0) 58%, rgba(20,20,26,0.05) 100%); }
  /* Le grain s'ajoute à la lumière de la tôle sans en changer la teinte ;
     l'usure s'y soustrait. D'où superposition pour l'un, produit pour
     l'autre. Chacun son SVG : un mix-blend-mode posé sur un <rect> à
     l'intérieur ne fusionne qu'avec le SVG, et la tôle derrière ne voit
     rien. */
  .film { position: absolute; inset: 0; width: 100%; height: 100%;
          pointer-events: none; z-index: 2; }
  /* 0,15 et 0,11, après 0,26 et 0,20, après 0,42 et 0,30 au banc d'essai.
     Les valeurs du banc étaient jugées sur une plaque seule et en grand. Six
     plaques de 104 px côte à côte dans une bande de 70, le grain s'additionne
     d'une plaque à l'autre et la rangée se lit comme sale.

     Rien ne fait varier ces deux valeurs à l'exécution : elles ne sont
     écrites nulle part ailleurs, et la graine du bruit est dérivée du rang de
     la plaque, donc stable d'un rendu à l'autre. Le seul moment où le motif
     change d'aspect est un changement de largeur de la fenêtre : la région du
     filtre est donnée en pourcentage, donc la turbulence est rééchantillonnée
     sur une autre boîte. */
  .grain { mix-blend-mode: overlay; opacity: var(--sign-grain, 0.15); }
  .wear { mix-blend-mode: multiply; opacity: var(--sign-wear, 0.11); }
  /* La flèche d'affectation de voie, en vectoriel : l'angle vit dans le path
     et l'élément ne porte que la translation. En masque CSS elle était une
     image tramée qu'on faisait tourner, nette à 90° et empâtée à 45°. */
  .lane { position: absolute; left: 50%; bottom: 6px; width: 19px;
          height: 19px; margin-left: -9.5px; z-index: 3; overflow: visible;
          fill: currentColor; opacity: 0; transform: translateY(-5px);
          transition: opacity 0.16s ease,
                      transform 0.26s cubic-bezier(0.22,0.7,0.3,1); }
  :host(:hover) .lane, :host(.on) .lane {
      opacity: 1; transform: translateY(0); }
  @media (prefers-reduced-motion: reduce) {
    .film, .ink { display: none; }
    .lane, :host(:hover) .lane, :host(.on) .lane {
        transition: opacity 0.16s ease; transform: none; } }
`;

/* La flèche d'affectation de voie est une pièce trapue, pas un chevron : sur
   un panneau réel la tête fait trois fois la largeur du fût et la moitié de
   la longueur totale. Le glyphe pointe à droite ; la rotation fait les
   voies. */
const ARROW_D = "M1 9.4h11V4L23 12l-11 8v-5.4H1z";

/* La norme n'emploie que deux angles : tout droit vers le bas, ou la
   bas-droite. La plaque en cours sort vers le bas, les autres en bas-droite,
   et les deux groupes ont donc la même paire quel que soit leur nombre de
   plaques. L'éventail à 135/90/45 faisait sortir Courses à 45° quand Heure
   sortait à 135°, en miroir, parce que les groupes n'ont pas le même nombre
   de plaques. */
function laneAngle(active) { return active ? 90 : 45; }

function laneSvg(deg) {
  return '<svg class="lane" viewBox="0 0 24 24" aria-hidden="true"'
    + ' focusable="false"><path d="' + ARROW_D + '"'
    + ' transform="rotate(' + deg + ' 12 12)"/></svg>';
}

/* Une nappe de bruit : un SVG, un filtre, un rectangle. L'identifiant du
   filtre porte la graine, sinon deux nappes du même document se partagent le
   même filtre et toutes les plaques prennent le grain de la première. */
function noiseLayer(cls, id, type, fq, oct, seed, table, mask) {
  const f = "f" + id;
  let out = '<svg class="' + cls + '" aria-hidden="true" focusable="false">'
    + '<filter id="' + f + '" x="0" y="0" width="100%" height="100%"'
    + ' color-interpolation-filters="sRGB">'
    + '<feTurbulence type="' + type + '" baseFrequency="' + fq + '"'
    + ' numOctaves="' + oct + '" seed="' + seed + '" result="n"/>'
    + '<feColorMatrix in="n" type="saturate" values="0"/>'
    + '<feComponentTransfer><feFuncA type="table" tableValues="' + table + '"/>'
    + "</feComponentTransfer></filter>";
  if (mask) {
    /* L'usure se retient sur les arêtes : une tôle s'use là où elle frotte et
       prend la pluie. Salie au centre, elle se lirait comme une tache. */
    out += '<radialGradient id="e' + id + '" cx="50%" cy="50%" r="62%">'
      + '<stop offset="0%" stop-color="#000"/>'
      + '<stop offset="58%" stop-color="#000"/>'
      + '<stop offset="100%" stop-color="#fff"/></radialGradient>'
      + '<mask id="m' + id + '"><rect width="100%" height="100%"'
      + ' fill="url(#e' + id + ')"/></mask>';
  }
  out += '<rect width="100%" height="100%" filter="url(#' + f + ')"'
    + (mask ? ' mask="url(#m' + id + ')"' : "") + "/></svg>";
  return out;
}

function plateFilm(seed) {
  return noiseLayer("film grain", "g" + seed, "fractalNoise", "0.8", 4,
                    seed, "0 0.75", false)
    + noiseLayer("film wear", "w" + seed, "fractalNoise", "0.045", 2,
                 seed + 97, "0 0 0.9", true);
}

/* Le grain qui passe au-dessus de l'encre. Plus fin et deux fois moins
   appuyé que celui de la tôle : on le lit sur les lettres, pas à travers. */
function inkFilm(seed) {
  return noiseLayer("film ink", "i" + seed, "fractalNoise", "1.1", 3,
                    seed + 53, "0 0.7", false);
}

class NycSign extends HTMLElement {
  connectedCallback() {
    if (this.dataset.ready) { return; }
    this.dataset.ready = "1";
    const seed = Number(this.dataset.seed || 0);
    const root = this.attachShadow({ mode: "open" });
    let html = "<style>" + SIGN_CSS + "</style>";
    /* La matière n'existe que de jour. De nuit il n'y a pas de plaque mais un
       phare : du grain sur une lumière n'aurait rien à grainer. */
    if (CFG.mode === "light") {
      html += '<div class="skin"><div class="sheen"></div>'
        + '<div class="ambient"></div>'
        + plateFilm(seed) + '<div class="edge"></div></div>';
    }
    html += '<span class="t"></span>';
    if (CFG.mode === "light") {
      html += laneSvg(Number(this.dataset.lane || 90))
        + '<div class="skin over">' + inkFilm(seed) + "</div>";
    }
    root.innerHTML = html;
    this.setAttribute("role", "button");
    if (!this.hasAttribute("tabindex")) { this.setAttribute("tabindex", "0"); }
    this.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
        e.preventDefault();
        this.click();
      }
    });
  }

  set label(v) {
    const t = this.shadowRoot && this.shadowRoot.querySelector(".t");
    if (t) { t.textContent = v; }
  }

  /* L'angle vit dans le path et non dans un transform CSS : c'est ce qui le
     garde net. Le changer veut donc dire réécrire l'attribut. */
  set lane(deg) {
    this.dataset.lane = String(deg);
    const path = this.shadowRoot
      && this.shadowRoot.querySelector(".lane path");
    if (path) { path.setAttribute("transform", "rotate(" + deg + " 12 12)"); }
  }
}
if (!customElements.get("nyc-sign")) {
  customElements.define("nyc-sign", NycSign);
}

/* ------------------------------------------------------------------
   L'armature, en trois temps : la forme, la matière, les ombres.

   Elle était dessinée pièce par pièce — une membrure, puis une diagonale,
   puis l'autre, chacune sa couleur. Chaque recouvrement se voyait, et à
   quatre pixels d'épaisseur ça ne pardonne pas. Ici les pièces ne servent
   qu'à définir une silhouette, posée une fois dans un masque ; tout ce qui
   suit est peint à travers elle, en rectangles pleins. Aucune pièce ne
   recouvre une autre, parce qu'il n'y a plus de pièces — il y a une
   charpente.
   ------------------------------------------------------------------ */
const TRUSS_H = 46;     /* hauteur de la poutre */
const CHORD = 5;        /* épaisseur d'une membrure */
const BAR = 4;          /* épaisseur d'une diagonale */

/* La perspective. On regarde le milieu de l'armature, donc le point de fuite
   est au centre de la poutre et la ferme du fond est cette même ferme réduite
   autour de ce point — en largeur comme en hauteur, d'un seul facteur. C'est
   la projection exacte : un point à la distance x du centre se projette en
   x·d/(d+t), où t est la profondeur du caisson et d la distance d'où on
   regarde. Une homothétie, donc, et pas deux échelles séparées.

   0,95 dit une profondeur de caisson d'un dix-neuvième de la distance
   d'observation, soit sept pixels d'écart apparent aux deux bouts. C'est peu,
   et c'est juste : une travée vue de loin est presque plate, et la profondeur
   se lit à ses entretoises bien plus qu'à l'écart des fermes.

   Sa maille compte un panneau de plus que celle de devant, pour que ses
   croisements tombent entre ceux du premier plan et se voient par les vides :
   à maille identique ils se superposaient exactement. */
const FAR_S = 0.95;

/* La lumière de la scène. Son rapport est deux de haut pour un de côté, le
   même que celui de toutes les ombres portées — une scène ne tient que si ses
   ombres partagent un seul rapport. */
const LIGHT = [-0.4472, -0.8944];

/* Une course de X posée où on veut : le plan du fond est la même ferme
   réduite, donc il lui faut une origine. */
function xRunAt(x0, y0, n, P, hTop, hBot) {
  let up = "", dn = "";
  for (let k = 0; k < n; k++) {
    const a = x0 + k * P, b = a + P;
    up += "M" + a.toFixed(2) + " " + (y0 + hBot).toFixed(2)
        + "L" + b.toFixed(2) + " " + (y0 + hTop).toFixed(2);
    dn += "M" + a.toFixed(2) + " " + (y0 + hTop).toFixed(2)
        + "L" + b.toFixed(2) + " " + (y0 + hBot).toFixed(2);
  }
  return [up, dn];
}

/* La géométrie d'une ferme, centrée sur la poutre. `kw` échelonne la portée
   et `k` la hauteur avec les épaisseurs. Elle est calculée en pixels réels et
   non posée par un `transform`, pour que l'ombrage des tubes travaille aux
   vraies épaisseurs.

   La diagonale va jusqu'à l'axe de la membrure et non jusqu'à sa face
   intérieure : arrêtée à la face, une barre de quatre pixels inclinée à
   quarante-cinq degrés n'y touche que par un coin, et il reste une encoche à
   la jonction. */
function ferme(w, kw, k, n) {
  const H = TRUSS_H;
  const hh = H * k, ww = w * kw, chord = CHORD * k, bar = BAR * k;
  const x0 = (w - ww) / 2, y0 = (H - hh) / 2;
  const P = ww / n;
  const [up, dn] = xRunAt(x0, y0, n, P, chord / 2, hh - chord / 2);
  const ch = "M" + x0.toFixed(2) + " " + (y0 + chord / 2).toFixed(2)
    + "L" + (x0 + ww).toFixed(2) + " " + (y0 + chord / 2).toFixed(2)
    + "M" + x0.toFixed(2) + " " + (y0 + hh - chord / 2).toFixed(2)
    + "L" + (x0 + ww).toFixed(2) + " " + (y0 + hh - chord / 2).toFixed(2);
  return { x0: x0, y0: y0, ww: ww, hh: hh, chord: chord, bar: bar,
           up: up, dn: dn, ch: ch };
}

function fermeShape(f) {
  return '<path d="' + f.ch + '" stroke="#fff" stroke-width="' + f.chord
    + '" fill="none"/>'
    + '<path d="' + (f.up + f.dn) + '" stroke="#fff" stroke-width="' + f.bar
    + '" fill="none"/>';
}

/* Les entretoises du caisson : les traverses qui relient la ferme de devant à
   celle du fond. Vues d'où l'on est, elles rayonnent du point de fuite —
   nulles au milieu de la poutre, où elles pointent droit sur nous, et les plus
   longues aux deux bouts. Ce sont elles, et rien d'autre, qui font lire un
   volume : sans elles les deux fermes ne sont que deux rectangles emboîtés.

   Elles ne sont pas rognées au puits, puisqu'elles joignent les axes des
   membrures, qui sont justement au bord de celui-ci. */
function entretoises(near, far) {
  const ay = near.chord / 2, by = near.hh - near.chord / 2;
  const fay = far.y0 + far.chord / 2, fby = far.y0 + far.hh - far.chord / 2;
  const seg = (x1, y1, x2, y2) => "M" + x1.toFixed(2) + " " + y1.toFixed(2)
    + "L" + x2.toFixed(2) + " " + y2.toFixed(2);
  return seg(near.x0, ay, far.x0, fay)
    + seg(near.x0, by, far.x0, fby)
    + seg(near.x0 + near.ww, ay, far.x0 + far.ww, fay)
    + seg(near.x0 + near.ww, by, far.x0 + far.ww, fby);
}

/* Le volume d'un tube, en trois passes décalées en travers de son fût.

   Ce qui compte pour un cylindre, c'est la part de la lumière perpendiculaire
   à son axe : un tube éclairé dans l'axe reste uni, un tube éclairé en
   travers a un bord sombre, un ventre clair et un rebond sur l'arête opposée.
   D'où trois ombrages différents issus d'une seule lumière — la membrure,
   horizontale, prend tout ; la diagonale montante, perpendiculaire à la
   lumière, presque tout ; la descendante, qui file dans son axe, deux fois
   moins. C'est cette inégalité qui fait lire trois tubes orientés et non un
   dessin plat.

   Le reflet est large et terne, pas un filet vif : le zinc d'une
   galvanisation à chaud est spangle et rugueux, donc sa tache spéculaire
   s'étale et s'éteint. Un acier brossé aurait le filet serré. */
function round3(d, wd, nx, ny, gain) {
  const k = Math.abs(nx * LIGHT[0] + ny * LIGHT[1])
    * (gain === undefined ? 1 : gain);
  const off = wd * 0.34;
  const pass = (dx, dy, col, op, shrink) =>
    '<path d="' + d + '" stroke="' + col + '" stroke-width="'
    + (wd - shrink).toFixed(2) + '" fill="none" opacity="'
    + (op * k).toFixed(3) + '" transform="translate(' + (dx * off).toFixed(2)
    + " " + (dy * off).toFixed(2) + ')"/>';
  return pass(-nx, -ny, "#000", 0.40, 0)
    + pass(nx, ny, "#ffffff", 0.30, wd * 0.30)
    + pass(nx * 1.7, ny * 1.7, "#ffffff", 0.20, wd * 0.56);
}

function fermeRelief(f, gain) {
  return round3(f.ch, f.chord, 0, -1, gain)
    + round3(f.up, f.bar, -0.707, -0.707, gain)
    + round3(f.dn, f.bar, 0.707, -0.707, gain);
}

function trussSvg(w, seed) {
  const H = TRUSS_H;
  const hi = H - 2 * CHORD;                 /* l'ouverture */
  /* Un panneau de croix de Saint-André est à peu près carré : sa longueur
     part de la hauteur de l'ouverture, puis s'ajuste à la portée pour que le
     compte soit entier. La longueur était figée et le reste coupé aux deux
     bouts — d'où des X qui finissaient dans le vide. */
  const n = Math.max(1, Math.round(w / hi));
  const near = ferme(w, 1, 1, n);
  const far = ferme(w, FAR_S, FAR_S, n + 1);
  const u = "t" + seed;
  const full = (fill, cls, filt) =>
    '<rect ' + (cls ? 'class="' + cls + '" ' : "")
    + 'width="' + w + '" height="' + H + '"'
    + (fill ? ' fill="' + fill + '"' : "")
    + (filt ? ' filter="url(#' + filt + u + ')"' : "") + "/>";
  const head = (cls) => '<svg class="beam ' + cls + '" width="' + w
    + '" height="' + H + '" viewBox="0 0 ' + w + " " + H
    + '" aria-hidden="true" focusable="false">';
  /* L'ambiance : le haut d'une charpente prend le ciel, le bas est dans son
     ombre propre. Ce n'est que la toile de fond — le volume des tubes est
     fait par fermeRelief(). */
  const amb = (id, a, b) => '<linearGradient id="' + id + u
    + '" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="' + H + '">'
    + '<stop offset="0%" stop-color="#fff" stop-opacity="' + a + '"/>'
    + '<stop offset="50%" stop-color="#000" stop-opacity="0"/>'
    + '<stop offset="100%" stop-color="#000" stop-opacity="' + b + '"/>'
    + "</linearGradient>";

  /* La ferme du fond, sur sa propre couche : elle est près du panneau, donc
     son ombre est serrée et franche. Pas de matière — on ne lit pas le grain
     d'une pièce qui n'est pas au premier plan. Et rognée au puits, l'intérieur
     du cadre de devant : vu de face on ne l'aperçoit que par là. */
  const back = head("far") + "<defs>"
    + '<clipPath id="w' + u + '"><rect x="0" y="' + CHORD + '" width="' + w
    + '" height="' + hi + '"/></clipPath>'
    + '<mask id="n' + u + '" maskUnits="userSpaceOnUse" x="0" y="0"'
    + ' width="' + w + '" height="' + H + '">' + fermeShape(far) + "</mask>"
    + amb("p", 0.12, 0.14)
    + "</defs>"
    + '<g clip-path="url(#w' + u + ')">'
    + '<g mask="url(#n' + u + ')">'
    + full("#74747e") + full("url(#p" + u + ")")
    + "</g>"
    + '<g mask="url(#n' + u + ')">' + fermeRelief(far, 0.7) + "</g>"
    + "</g>"
    + '<path d="' + entretoises(near, far) + '" stroke="#5e5e68"'
    + ' stroke-width="' + (far.bar * 1.05).toFixed(2) + '" fill="none"/>'
    + '<path d="' + entretoises(near, far) + '" stroke="#9a9aa4"'
    + ' stroke-width="' + (far.bar * 0.5).toFixed(2) + '" fill="none"/>'
    + "</svg>";

  /* La ferme de devant : loin du panneau, donc son ombre est large, molle et
     pâle. Forme, matière, ombres, dans cet ordre, toutes peintes à travers la
     même silhouette. */
  const front = head("near") + "<defs>"
    + '<mask id="m' + u + '" maskUnits="userSpaceOnUse" x="0" y="0"'
    + ' width="' + w + '" height="' + H + '">' + fermeShape(near) + "</mask>"
    /* L'acier : une seule rampe pour toute la charpente. La lumière vient
       d'en haut, donc la membrure haute est claire et la basse sombre. Un
       dégradé par pièce aurait rendu les jonctions. */
    + '<linearGradient id="a' + u + '" gradientUnits="userSpaceOnUse"'
    + ' x1="0" y1="0" x2="0" y2="' + H + '">'
    + '<stop offset="0%" stop-color="#dcdce4"/>'
    + '<stop offset="30%" stop-color="#bcbcc6"/>'
    + '<stop offset="68%" stop-color="#9696a0"/>'
    + '<stop offset="100%" stop-color="#6c6c76"/></linearGradient>'
    + amb("o", 0.16, 0.12)
    /* La matière, en trois nappes, et c'est le type de bruit qui compte :

       1. Le spangle. La galvanisation à chaud fait cristalliser le zinc en
          plaques irrégulières de quelques millimètres, visibles à l'œil nu.
          C'est `type="turbulence"` qu'il faut, pas `fractalNoise` : le premier
          donne des cellules, le second un nuage lisse. Et la courbe de
          transfert est raide, pour que les plaques aient un bord.
       2. Les stries de laminage, serrées dans l'axe de la pièce.
       3. L'encrassement : très basse fréquence, en produit. Un profilé posé
          dehors n'a pas la même valeur partout, et c'est cette inégalité-là,
          plus que le grain, qui enlève l'air neuf. */
    + trussFilter("s" + u, "turbulence", "0.5", 2, seed, "0 0.15 0.9 1")
    + trussFilter("d" + u, "fractalNoise", "0.02 0.8", 2, seed + 41, "0.15 0.9")
    + trussFilter("g" + u, "fractalNoise", "0.045 0.03", 3, seed + 77, "0.35 1")
    + "</defs>"
    /* isolation: isolate — sans elle les nappes fusionneraient avec la page et
       non avec l'acier posé juste en dessous. */
    + '<g mask="url(#m' + u + ')" style="isolation: isolate">'
    + full("url(#a" + u + ")")
    + full(null, "soil", "g") + full(null, "spangle", "s")
    + full(null, "draw", "d")
    + full("url(#o" + u + ")")
    + fermeRelief(near, 1)
    + "</g></svg>";

  return back + front;
}

function trussFilter(id, type, fq, oct, seed, table) {
  return '<filter id="' + id + '" x="0" y="0" width="100%" height="100%"'
    + ' color-interpolation-filters="sRGB">'
    + '<feTurbulence type="' + type + '" baseFrequency="' + fq + '"'
    + ' numOctaves="' + oct + '" seed="' + seed + '" result="n"/>'
    + '<feColorMatrix in="n" type="saturate" values="0"/>'
    + '<feComponentTransfer><feFuncA type="table" tableValues="' + table + '"/>'
    + "</feComponentTransfer></filter>";
}

/* La poutre ne peut être dessinée qu'une fois sa largeur connue, et elle
   change avec celle de la fenêtre : d'où le ResizeObserver, qui la redessine
   au pixel près. Elle est recréée à chaque peinture des boutons, puisque
   celle-ci vide le groupe. */
const TRUSS_SEED = { measures: 13, scopes: 31 };

function fitTruss(set) {
  if (CFG.mode !== "light") { return; }
  let t = set.querySelector(".truss");
  if (!t) {
    t = document.createElement("i");
    t.className = "truss";
    set.insertBefore(t, set.firstChild);
  }
  const w = Math.round(set.getBoundingClientRect().width);
  if (w > 0 && w !== t._w) {
    t._w = w;
    t.innerHTML = trussSvg(w, TRUSS_SEED[set.id] || 13);
  }
}

let timer = null;

function sign(label, active, i, n, onclick) {
  const b = document.createElement("nyc-sign");
  b.className = "sign" + (active ? " on" : "");
  b.dataset.lane = String(laneAngle(active));
  /* Une graine par plaque et par groupe : deux plaques au même grain se
     voient tout de suite, surtout côte à côte. */
  b.dataset.seed = String(i * 7 + n * 31);
  b.setAttribute("aria-pressed", active ? "true" : "false");
  /* connectedCallback construit la racine fantôme, et il ne tourne qu'une
     fois l'élément dans le document. Le mot est donc posé après insertion,
     par le compte rendu du label, et pas ici. */
  b._label = label;
  b.onclick = onclick;
  return b;
}

function paintButtons() {
  const sc = document.getElementById("scopes");
  sc.innerHTML = "";
  const scopes = Object.keys(META.cycles);
  scopes.forEach((k, i) => {
    const el = sign(META.cycles[k].label, S.scope === k, i, scopes.length,
                    () => {
      S.scope = k; S.step = 0; S.at = Date.now(); publish();
    });
    sc.appendChild(el);
    el.label = el._label;
  });
  const me = document.getElementById("measures");
  me.innerHTML = "";
  const order = measuresInOrder();
  order.forEach((m, i) => {
    const el = sign(MEASURE_LABEL[m], S.measure === m, i, order.length,
                    () => {
      S.measure = m; publish();
    });
    me.appendChild(el);
    el.label = el._label;
  });
  /* Le pas courant est écrit en clair : le ruban le montre aussi, mais le
     lire ici évite de chercher quelle barre porte l'accent.
     Le nom entier et non l'abrégé — « 18h », « Mardi 18h », « Juillet 18h ».
     Depuis que la semaine et l'année portent l'heure, l'abrégé du ruban ne
     dit plus que l'heure sur vingt-trois pas sur vingt-quatre : il faut le
     nom du jour ou du mois ici, sans quoi on ne sait plus où l'on est. */
  document.getElementById("now").textContent =
    META.cycles[S.scope].labels[S.step];
  document.getElementById("play").textContent = S.playing ? "❚❚" : "▶";
  /* Vider un groupe emporte sa poutre : on la repose après les plaques, et
     elle se mesure elle-même. */
  fitTruss(me);
  fitTruss(sc);
}

function publish() {
  channel.postMessage({ type: "state", state: S });
  paintButtons();
  schedule();
}

/* Un minuteur plutôt qu'une boucle d'animation : ce composant ne dessine
   rien en continu, il ne fait qu'avancer le pas. Les autres interpolent. */
function schedule() {
  clearTimeout(timer);
  if (!S.playing) { return; }
  timer = setTimeout(() => {
    S.step = (S.step + 1) % stepsOf(S.scope);
    S.at = Date.now();
    publish();
  }, META.cycles[S.scope].ms);
}

function boot() {
  S.scope = CFG.scope; S.measure = CFG.measure;
  const p = document.getElementById("play");
  p.onclick = () => {
    S.playing = !S.playing;
    S.at = Date.now();
    publish();
  };
  publish();
  /* La poutre se dessine en pixels mesurés, donc elle doit être redessinée
     quand la largeur change. Un observateur par groupe plutôt que
     l'évènement `resize` de la fenêtre : au premier passage l'iframe peut
     n'avoir aucune dimension, et l'observateur part dès qu'une largeur
     réelle existe. La poutre est en position absolue, donc la poser ne
     change pas la taille du groupe et l'observateur ne se rappelle pas
     lui-même. */
  if (window.ResizeObserver) {
    const o = new ResizeObserver(es => { es.forEach(e => fitTruss(e.target)); });
    o.observe(document.getElementById("measures"));
    o.observe(document.getElementById("scopes"));
  }
}

/* Le maître de l'horloge répond aux arrivants, et aux demandes de saut. */
channel.addEventListener("message", e => {
  if (!e.data || !META) { return; }
  if (e.data.type === "hello") { publish(); }
  if (e.data.type === "seek") {
    const n = stepsOf(S.scope);
    S.step = ((e.data.step % n) + n) % n;
    /* L'horloge repart du pas demandé sans s'arrêter : un saut est un
       déplacement, pas une pause. Le bouton ❚❚ reste là pour s'arrêter. */
    S.at = Date.now();
    publish();
  }
});
</script>
"""

# ---------------------------------------------------------------------------
# Composant 1 bis — les deux cadrans du pas courant
# ---------------------------------------------------------------------------
_KPIS = r"""
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  /* La planche de bord, en quatre quartiers :

       1 vitesse    2 tarif
       3 passagers  4 distance

     L'ordre est celui du regard d'un conducteur : le compteur de vitesse
     devant lui, le taximètre à sa droite sur la console, le volant sous le
     compteur, l'écran du GPS sous le taximètre. Chaque case est un objet de
     l'habitacle à sa place, et non une case de tableau.

     Deux colonnes et non une : les quatre objets sont compacts — un cadran,
     un afficheur, un demi-volant, un écran — et empilés sur une colonne ils
     n'auraient eu que soixante pixels de haut chacun. */
  #grid { position: absolute; inset: 0; display: grid;
          grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
          gap: 6px 16px; }
  /* Trois rangs : le titre, le dessin, le sous-titre. Les quatre cases les
     posent aux mêmes hauteurs, donc les titres des deux cases d'un même rang
     s'alignent — en pile centrée, ils flottaient chacun selon la hauteur de
     son dessin. Et le dessin est centré dans sa case au lieu d'être collé à
     gauche : un instrument est au milieu de son cadran de bord. */
  /* Deux rangs : le titre, le dessin. Le sous-titre est parti — il disait la
     moyenne du cycle, le dollar par mile, la base de la répartition. Trois
     phrases de onze pixels sous quatre instruments : chacune cassait la
     lecture de celui qu'elle légendait, et aucune ne se lisait. Les quatre
     cases se lisent maintenant comme quatre cadrans.

     Ce que la base des passagers disait — 15,7 % des courses ne déclarent
     rien — n'est plus dit nulle part dans le bloc. C'est le prix du silence,
     et il est assumé : la répartition est juste, c'est son assiette qui est
     partielle. */
  .kpi { display: grid; grid-template-rows: auto 1fr;
         justify-items: center; align-items: center;
         min-width: 0; text-align: center; }

  /* Quatre traits et non deux : deux traits entiers se croiseraient au
     milieu, et une croix est un cadre. Quatre segments qui s'arrêtent avant
     le centre séparent sans enfermer. Chacun s'éteint à ses deux bouts —
     un trait net aurait été une règle, pas une séparation. */
  .sep { position: absolute; pointer-events: none; }
  .sep.v { width: 1px; left: 50%; }
  .sep.h { height: 1px; }
  .sep.v1 { top: 7%; height: 36%; }
  .sep.v2 { top: 57%; height: 36%; }
  .sep.h1 { top: 50%; left: 5%; width: 40%; }
  .sep.h2 { top: 50%; left: 55%; width: 40%; }
  .k-lab { font-size: 0.64rem; letter-spacing: 0.09em;
           text-transform: uppercase; }
  .k-val { font-size: 1.15rem; font-weight: 600; line-height: 1.25;
           font-variant-numeric: tabular-nums; }
  .k-wide { display: flex; align-items: center; gap: 11px; min-width: 0; }

  /* La répartition des passagers : le volant, vu comme on le voit en
     conduisant — la moitié haute de la jante, le moyeu coupé par le bas du
     cadre. Un volant entier posé au milieu d'une case est une roue ; coupé
     par le bas, c'est un volant.

     Deux parts et non trois : « un passager » contre « deux et plus ». À
     trois, les deux petites prenaient 14 % et 6 % de la jante, soit un arc de
     25 et un de 11 degrés sur un demi-tour — trop court pour porter un dessin
     au-dessus de lui. Les séries des deux et des trois-et-plus sont sommées
     ici. */
  .k-wheel { position: relative; flex: 0 0 auto;
             width: 200px; height: 100px; }
  .k-3d  { position: absolute; left: 0; top: 0; width: 200px; height: 100px;
           z-index: 1; }
  .k-rim { position: absolute; left: 0; top: 0; width: 200px; height: 100px;
           z-index: 2; }
  /* L'étiquette d'une part : le dessin des passagers, et dessous la part en
     pour cent. Elle est posée au milieu angulaire de son arc, sur une
     ellipse hors de la jante — une ellipse et non un cercle, la case étant
     deux fois plus large que haute, et un rayon constant sortait du cadre en
     haut et en bas.

     Elle suit sa part : quand l'arc tourne, l'étiquette tourne avec lui, sur
     la même durée. En HTML et non en SVG, parce qu'une transition sur un
     `transform` d'élément HTML est tenue par tous les navigateurs quand
     celle d'un noeud SVG ne l'est pas partout. */
  .k-wheel .tag { position: absolute; left: 0; top: 0;
                  display: flex; flex-direction: column; align-items: center;
                  gap: 1px; line-height: 1.1; white-space: nowrap;
                  transition: transform var(--arc, 0.42s) cubic-bezier(0, 0, 0.3, 1); }
  .k-wheel .kt { display: flex; align-items: flex-end; gap: 1px; }
  .k-wheel .fig { width: 7px; height: 10px; fill: currentColor;
                  flex: 0 0 auto; }
  .k-wheel .plus { font-weight: 700; font-size: 0.72rem; margin-left: 1px; }
  .k-wheel .kv { font-size: 0.72rem; font-weight: 700;
                 font-variant-numeric: tabular-nums; }

  /* Le cadran de vitesse — un compteur de voiture, aiguille au centre et
     course de 240 degrés, pas un demi-cercle. Graduations à l'intérieur de
     l'arc, les deux bornes du cycle chiffrées en pied de cadran, et un
     repère fin sur la moyenne du cycle : sans lui, l'aiguille dit une
     vitesse mais pas si elle est haute.

     Comme l'anneau, le remplissage est un dasharray et l'aiguille une
     rotation : le navigateur interpole les deux, aucune boucle
     d'animation en JavaScript ne s'en occupe. */
  /* Le cadran et le volant font 200 px de large pour 72 et 64 de dessin :
     leurs axes tombent alors au même endroit, à 100 px du bord de la case, et
     les deux instruments sont sur une colonne. Le reste de la largeur est la
     place des étiquettes, qui sortent du cercle des deux côtés. */
  .k-dial { flex: 0 0 auto; width: 200px; height: 96px; }
  .k-dial .ndl { transition: transform var(--arc, 0.42s) cubic-bezier(0, 0, 0.3, 1); }
  /* L'écran à cristaux liquides du cadran. L'encre #1f3413 est relevée au
     pixel sur la photo de référence : un olive très sombre, jamais un noir.
     Les segments éteints sont la même encre à onze pour cent, ce qui est
     leur contraste réel sur un afficheur rétro-éclairé. L'annonciateur est
     du texte plat, comme le sin et le cos d'une calculatrice.

     Ces trois règles ne suivent pas le thème : un afficheur vert est vert
     le jour comme la nuit. */
  .k-dial .seg-on { fill: #1f3413; }
  .k-dial .seg-off { fill: #1f3413; opacity: 0.11; }
  .k-dial .lcd-ann { fill: #1f3413; opacity: 0.78; font-size: 3.4px;
                     font-weight: 500; letter-spacing: 0.12em; }
  /* Les nombres d'un cadran de voiture sont gros : ils sont la graduation,
     les traits ne sont que leur subdivision. À 10,5 px pour un cadran de 80
     de diamètre, ils faisaient une légende ; à 12,5, ils font le cadran.

     Et il y en a un tous les cinq et non tous les dix. Un compteur de série
     en porte une douzaine sur sa course — 0 à 240 par 20 — là où trois
     nombres sur 240 degrés laissaient le cadran vide entre eux. */
  /* Les nombres tiennent leur propre couronne, à 28 du centre, et non plus
     le pied des graduations. Tant qu'ils étaient posés sur les traits, les
     deux se gênaient : il fallait les faire gros pour qu'ils se détachent,
     et gros ils mangeaient la graduation. Séparés, ils peuvent être fins. */
  .k-dial .bound { font-size: 10px; font-weight: 700;
                   letter-spacing: -0.02em;
                   font-variant-numeric: tabular-nums; }

  /* Le taximètre, relevé sur l'appareil de référence et non deviné.

     Les couleurs sont échantillonnées sur la photo : les chiffres sont
     blancs #ffffff, les étiquettes d'un cyan vif #70d5ff, et la vitre est un
     gris moyen — médiane #888888 à gauche, #9c9c9c à droite, donc un
     dégradé. Pas de jaune nulle part : c'était mon invention.

     Le contraste de l'objet réel est faible et c'est voulu, puisque la vitre
     est une glace : le blanc sur ce gris donne 3,1 à 3,9 contre 1, et le
     cyan 2,2 à 2,6. Les chiffres font 38 px en grand et 20 px en petit, donc
     au-dessus de 18,66 px où le minimum tombe à 3 contre 1 : ils passent.
     Les étiquettes, non — elles sont sous le seuil, comme sur l'appareil.
     C'est un écart assumé, choisi pour ressembler à l'objet, et aucune
     étiquette ne porte de chiffre.

     Pas de boîtier et pas de contour : la vitre seule, posée sur la case.
     Le bloc porte déjà quatre instruments, et un cadre de plus autour de
     celui-ci en faisait un objet rapporté.

     La vitre est un gris moyen — #969696 à #868686 — et les chiffres tirent
     sur le jaune chaud, #fff2c4. Le contraste vaut alors 2,7 à 3,2 contre 1,
     donc au seuil des grands caractères, que les deux tailles de chiffres
     dépassent. À #b0b0b0, la vitre claire d'un premier essai, les mêmes
     chiffres tombaient à 1,9 : c'est le gris de la vitre qui tient la
     lisibilité, pas la couleur des chiffres. */
  /* L'étiquette est un bleu sombre et non le cyan vif de l'appareil : sur
     cette vitre grise, #1f8fc4 tombait à 1,0 contre 1 — exactement la
     clarté du fond, donc invisible. Le même bleu assombri tient 3,3 à 4,0. */
  .k-seg { --s-lit: #fff2c4; --s-off: rgb(255 255 255 / 0.15);
           --s-lab: #0a3a57; }
  /* Le look rétroviseur — la glace grise et les chiffres jaune pâle — est
     celui du thème clair. De nuit l'instrument redevient ce qu'il est dans
     une voiture : un afficheur rouge sur du verre noir. Le rouge-orangé est
     celui des compteurs montés au rétroviseur, et il ne sert qu'ici et à
     l'aiguille du compteur de vitesse. */
  .k-seg.night { --s-lit: #ff4326; --s-off: #3a1a12; --s-lab: #8d7a6c; }
  .k-seg .m-win { background: linear-gradient(170deg, #969696 0%,
                                              #868686 100%);
                  border-radius: 6px; padding: 6px 13px 5px;
                  display: flex; flex-direction: column;
                  align-items: center; gap: 2px; }
  .k-seg.night .m-win { background: linear-gradient(170deg, #16100e 0%,
                                                    #0b0807 100%); }
  .k-seg .m-seg { display: block; height: 38px; }
  .k-seg .m-sm { display: block; height: 20px; }
  .k-seg .off { fill: var(--s-off); }
  /* Les chiffres débordent un peu sur la vitre, comme sur l'appareil. */
  .k-seg .lit { fill: var(--s-lit);
                filter: drop-shadow(0 0 3px rgb(255 242 196 / 0.45)); }
  .k-seg.night .lit { filter: drop-shadow(0 0 6px rgb(255 67 38 / 0.3)); }
  /* La disposition de l'appareil : l'étiquette du grand nombre est à sa
     gauche, celles des petits sont dessous. Ce n'est pas un caprice de
     copiste — un nombre de tête se lit de gauche à droite avec son nom
     devant, et deux nombres côte à côte ont besoin que leur nom soit
     exactement sous eux pour ne pas s'échanger. */
  .k-seg .m-top { display: flex; align-items: center; gap: 8px; }
  .k-seg .m-row { display: flex; align-items: flex-start; gap: 16px; }
  .k-seg .m-one { display: flex; flex-direction: column; align-items: center;
                  gap: 1px; }
  .k-seg .lb { font-size: 0.54rem; font-weight: 700; letter-spacing: 0.05em;
               text-transform: uppercase; color: var(--s-lab);
               white-space: nowrap; }

  /* L'écran de navigation : une maquette de quartier new-yorkais vue depuis
     l'avenue, et par-dessus un panneau de verre qui porte les deux nombres.

     La boussole est partie. Elle encodait la distance dans son cap, ce qui
     demandait de savoir qu'un cap était une distance ; et une rose des vents
     n'est pas ce qu'on regarde dans un taxi, c'est un écran. La distance est
     maintenant dans le panneau, en chiffres, et dans la longueur du ruban
     d'itinéraire, qui court d'autant plus loin devant que la course est
     longue.

     La maquette est blanche dans les deux thèmes, comme le volant est noir
     dans les deux : c'est un objet de l'habitacle, pas une surface de la
     page. */
  .k-nav { position: relative; flex: 0 0 auto; width: 200px; height: 112px;
           border-radius: 7px; overflow: hidden; }
  .k-nav canvas { display: block; width: 100%; height: 100%; }
  /* Le panneau de verre. Le flou d'arrière-plan est ce qui le fait flotter :
     sans lui, c'est une boîte posée sur une image, et le fond repasse alors
     en opaque — mieux vaut perdre l'effet que la lisibilité. */
  .k-stack { position: absolute; left: 6px; top: 6px; width: 92px;
             border-radius: 8px; overflow: hidden;
             backdrop-filter: blur(9px) saturate(1.1);
             -webkit-backdrop-filter: blur(9px) saturate(1.1); }
  .k-stack .row { display: flex; align-items: center; gap: 6px;
                  padding: 5px 7px; }
  .k-stack svg { flex: none; width: 12px; height: 12px; fill: none;
                 stroke-width: 1.8; stroke-linecap: round;
                 stroke-linejoin: round; }
  .k-stack .v { font-size: 0.8rem; font-weight: 700; line-height: 1.05;
                letter-spacing: -0.02em;
                font-variant-numeric: tabular-nums; white-space: nowrap; }
  .k-stack .v i { font-style: normal; font-size: 0.5rem; font-weight: 600;
                  margin-left: 2px; }
  .k-stack .k { font-size: 0.5rem; font-weight: 500; letter-spacing: 0.06em;
                text-transform: uppercase; line-height: 1.3; }

  /* Le cadran. Le chiffre n'orbite plus autour de l'aiguille : il est dans
     l'écran à cristaux liquides posé sous l'axe. Un nombre qui tourne autour
     d'un cadran est une trouvaille ; un nombre dans une fenêtre est ce qu'on
     lit sur un compteur. */
  .k-gauge { position: relative; width: 200px; height: 96px; }

  /* Le repère de la part tourne avec elle, comme l'aiguille du compteur
     tourne avec la vitesse. C'est un attribut de présentation SVG, donc la
     transition le prend. */
  .k-rim .pt { transition: transform var(--arc, 0.42s) cubic-bezier(0, 0, 0.3, 1); }

  @media (prefers-reduced-motion: reduce) {
      .k-rim .pt, .k-wheel .tag,
      .k-dial .ndl { transition: none; } }
  /* Chaque valeur est une case où deux textes se superposent, le sortant et
     l'entrant : ils occupent la même ligne et se croisent sur place. Un seul
     élément qu'on vide puis remplit laisserait une case vide entre les deux,
     et c'est ce trou qui se voit, pas le changement. */
  .k-lab, .k-val { display: grid; min-width: 0; }
  .k-lab > span, .k-val > span {
      grid-area: 1 / 1; min-width: 0;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      transition: opacity 0.22s ease, transform 0.22s ease; }
  /* Le sortant monte, l'entrant vient d'en dessous : le même sens que le
     ruban, où le passé file et l'à-venir arrive. */
  .k-lab > span.out, .k-val > span.out {
      opacity: 0; transform: translateY(-0.3em); }
  .k-lab > span.in, .k-val > span.in {
      opacity: 0; transform: translateY(0.3em); }
  @media (prefers-reduced-motion: reduce) {
      .k-lab > span, .k-val > span { transition: opacity 0.22s ease; }
      .k-lab > span.out, .k-val > span.out,
      .k-lab > span.in, .k-val > span.in { transform: none; } }
</style>
<div id="grid"></div>
<script>
/* Un nombre va à l'autre en comptant, il ne se fond pas. Deux pas voisins
   sont voisins en valeur — 4 454 courses puis 5 065, jamais 2 000 puis
   20 000 — et un compteur rend cette proximité lisible : on voit la pente,
   pas seulement l'arrivée. C'est la même interpolation que la carte et le
   ruban, qui montent et glissent entre deux pas ; les nombres s'y mettent.

   Ce qui reste interdit, c'est le défilement continu : un chiffre qui bouge
   sans arrêt devient une texture. Le compteur court 420 ms puis se tait, et
   le pas en dure au moins 1 700 — la valeur passe donc les trois quarts du
   temps immobile, assez pour être lue.

   Les quatre cases partent en file, 70 ms d'écart. Quatre nombres qui
   changent exactement ensemble font un seul sursaut ; décalés, ils font une
   phrase, et l'œil les prend l'un après l'autre.

   Le nom de zone, lui, n'est pas un nombre : il ne compte pas, il se croise
   sur place. */
/* Deux cases, une seule règle : elles doivent bouger dans le cycle. Un
   nombre qui ne bouge pas n'appartient pas au cockpit, il appartient à la
   fiche signalétique — et la fiche est maintenant en pied de page.

   Les titres sont ceux d'une légende, pas d'un article : ils disent ce qui
   est montré et rien de plus. L'interprétation, si elle vient un jour, sera
   une phrase à part, pas un titre déguisé en conclusion.

   Mesuré sur les 24 pas de l'heure :
     — un seul passager, 88,9 % des courses à 6 h, 74,4 % à 22 h
     — deux passagers, 7,4 % à 6 h et 18,2 % à 22 h               (x2,5)
     — plus de deux, 3,7 % à 6 h et 8,8 % à 2 h                   (x2,4)
     — la vitesse, 10,1 mph à 17 h et 21,0 mph à 5 h              (x2,1)
     — la durée, 13,4 min à 2 h et 20,4 min à 15 h                (x1,5)
     — la carte sans pourboire, 3,8 % à 21 h et 13,9 % à 5 h      (x3,7)

   La première case porte une répartition et non un nombre, parce que ses
   parts ne culminent pas au même pas : deux passagers le soir, plus de deux
   en pleine nuit. Une part seule perdrait le décalage.

   Elle exclut de son dénominateur les courses qui ne déclarent rien :
   15,66 % des lignes, avec un taux d'absence qui suit l'heure — 9,9 % à
   16 h, 39,6 % à 4 h. Au dénominateur, la case aurait raconté son trou.

   La seconde porte la vitesse de la ville, en compteur. C'est un rapport et
   non un compte, donc un autre genre de mesure que l'anneau d'au-dessus :
   les deux cases ne se redisent pas. Et c'est le seul chiffre du cockpit
   qu'un lecteur peut vérifier contre son propre souvenir de la ville — elle
   roule à 21,0 mph à 5 h et à 10,1 mph à 17 h, et s'effondre de 11 h à 18 h
   sans se relever.

   Mesuré sur les trois cycles : x2,08 sur la journée, x2,62 sur la semaine
   — 9,06 mph un jeudi à 17 h contre 23,75 mph un dimanche à 6 h — et x1,21
   seulement sur l'année, qui est le cycle où rien ne bouge beaucoup.

   Les deux bornes de l'arc sont celles du cycle affiché, pas des bornes
   fixes : un compteur gradué de 0 à 30 aurait laissé l'aiguille dans le
   tiers du bas en permanence, et l'amplitude réelle n'aurait pas été
   lisible.

   Le sous-titre dit les deux termes du rapport, la distance et la durée,
   qui bougent tous les deux. L'arc dit « à quelle vitesse », les deux
   nombres disent « parce que ». La division est exacte, le payload portant
   mph = dist / (dur / 60).

   Écarté : l'encaissé total, corrélé aux courses à 0,995, qui les redit en
   dollars ; la zone la plus active, qui est Midtown vingt-deux pas sur
   vingt-quatre ; le nombre moyen de passagers, qui ne va que de 1,18 à
   1,39 et reste donc une case figée ; les minutes par mile, inverse exact
   de la vitesse à -0,98, qui feraient deux cases pour un seul nombre.

   Écarté aussi, et de peu : la part des courses qui touchent un aéroport,
   qui va de 3,3 % à 2 h à 21,9 % à 5 h, soit x6,63 — la plus forte
   amplitude de tout ce qui a été mesuré. Une course sur cinq à cinq heures
   du matin part de JFK ou de LaGuardia ou y va. Elle reste sur l'étagère
   parce que son dénominateur s'effondre aux heures creuses, là même où elle
   culmine : le fait est réel, mais une part qui monte quand son assiette
   descend demande une phrase, pas une case.

   La case liée au panneau allumé est partie avec ce changement. Elle
   affichait la hauteur, en chiffres, de la barre allumée dans le ruban
   d'à côté — donc le même nombre, deux fois, à trente centimètres près. Le
   bloc des nombres ne répond plus au sélecteur de mesure ; la carte, le
   ruban et les flux, eux, y répondent toujours. */
const CELLS = ["mph", "fare", "pax", "dist"];
/* Des étiquettes de légende : ce qui est mesuré, et son unité quand elle ne
   se lit pas dans le chiffre. « Nombre de passagers » ne disait pas par
   quoi on divisait, et « Vitesse de la ville » interprétait au lieu de
   nommer. */
/* L'unité est dans le titre, pas collée au nombre. Pour une valeur unique —
   un compteur, un afficheur — c'est la règle : le nombre reste nu et l'unité
   est dite une fois, à l'endroit où on va la chercher. Répéter « mph » et
   « $ » sur un chiffre qui change à chaque pas ajoute du bruit sans rien
   dire de plus. Sur un axe gradué, les deux usages se valent et le suffixe
   par graduation est courant ; ici il n'y a pas d'axe, il y a quatre
   valeurs.

   Le pour-cent fait exception et reste sur le volant : ce n'est pas une
   unité, c'est la marque d'une proportion, et deux nombres qui font un
   ensemble se lisent mal nus. */
const CELL_LABEL = { mph: "Vitesse moyenne (mph)",
                     fare: "Tarif moyen de la course ($)",
                     pax: "Passagers par course",
                     dist: "Temps et distance de la course" };

/* Un bonhomme, deux bonshommes, deux bonshommes et un plus. Le dessin dit ce
   que « 1 / 2 / 3+ » demandait de deviner, et il tient dans la hauteur d'une
   ligne de légende. */
const FIG = '<svg class="fig" viewBox="0 0 8 12" aria-hidden="true">'
  + '<circle cx="4" cy="2.9" r="2.3"/>'
  + '<path d="M0.5 12V9.5a3.5 3.5 0 0 1 7 0V12z"/></svg>';

/* Les deux répartitions. `k` est la série dans META.city, `fig` l'étiquette
   dessinée, `txt` celle que lisent les lecteurs d'écran et les légendes
   textuelles. L'ordre est celui de l'empilement et des arcs. */
const PARTS = {
  pax: [
    /* Une rampe d'ambre, du plus clair au plus foncé : « un, deux, plus de
       deux » est un ordre, et une rampe le dit là où trois teintes sans
       rapport le feraient lire comme trois catégories.

       Les trois valeurs sont fixes et non prises sur les jetons du thème.
       Le fond des bandes n'est pas le panneau de la page : c'est le creux du
       volant, le même charbon #15161a de jour comme de nuit. Une rampe qui
       suivrait le thème changerait de fond sans changer de fond.

       La rampe a été retournée, et c'est la correction qui compte. Elle
       allait du clair au foncé dans l'ordre un, deux, plus de deux — donc
       dans le sens où les arcs rétrécissent. La part de 76 % recevait le
       plus de contraste et celle de 7 % le moins : le plus petit objet du
       dessin était le plus difficile à voir. Elle va maintenant du foncé au
       clair, et le contraste contre le creux charbon suit la rareté — 5,0
       pour la part qui occupe les trois quarts de la jante, 15,5 pour celle
       qui n'en occupe qu'un quatorzième.

       C'est toujours une rampe d'une seule teinte, monotone en clarté, donc
       elle dit toujours un ordre — simplement dans l'autre sens.

       Mesuré : séparation 17,4 et 17,1 en DeltaE OKLab entre parts
       voisines, au-dessus du plancher de 15 en vision normale comme en
       deutan (16,6) et en tritan (16,0). */
    { k: ["solo"], fig: FIG, txt: "un passager",
      col: "#b07d1c", ink: "#432d05" },
    { k: ["duo"], fig: FIG + FIG, txt: "deux passagers",
      col: "#e8b44e", ink: "#7a520f" },
    { k: ["trio"], fig: FIG + FIG + '<b class="plus">+</b>',
      txt: "plus de deux", col: "#fbeccb", ink: "#a06d18" },
  ],
};

/* La valeur d'une part : la somme de ses séries. « Deux et plus » n'a pas de
   série à lui dans le payload, et n'en a pas besoin — les parts sont prises
   sur la même assiette, donc elles s'ajoutent. */
/* La couleur d'une part dans le creux du volant : la sienne si elle en a
   une, sinon un degré de la rampe grise. */
function partCol(p) { return p.col || tone(p.tone); }

/* La même part, mais posée sur le panneau de la page : le bonhomme de son
   étiquette et son repère sur la jante. Le fond n'est plus le charbon du
   creux mais le panneau, et sur un panneau blanc les deux pas clairs de la
   rampe tombaient à 1,3 et 2,1 contre 1 — le pas le plus clair était celui
   qu'on ne voyait pas. De nuit la rampe tient telle quelle ; de jour
   l'étiquette prend trois pas plus sombres de la même teinte, à 13,0, 6,9
   et 4,5 contre 1, donc toutes les trois au-dessus du plancher de 4,5, ce
   que l'ancienne encre ne faisait pas.

   Leur séparation tombe à 15,9 et 10,5 en DeltaE, sous le plancher de 15
   pour la seconde paire. C'est admis ici et seulement ici : l'étiquette
   porte un, deux ou trois bonshommes, donc l'identité de la part est dite
   par le dessin et la couleur ne fait que la confirmer. */
function partInk(p) {
  if (!p.ink) { return partCol(p); }
  return NIGHT ? partCol(p) : p.ink;
}


function partVal(part, step) {
  let v = 0;
  for (const k of part.k) { v += atStep(k, step); }
  return v;
}

/* Un degré de la rampe grise, pour une part qui ne porte pas sa couleur.
   tone = -1 sort de la rampe, d'où la brique.

   L'ambre avait été écarté quand il était la couleur de tout — la jante, les
   fenêtres des tours, la flèche. Il revient sur les parts du volant parce
   que la jante, elle, est devenue charbon : l'ambre n'y est plus la couleur
   du fond, il est la seule chose allumée dans le creux. */
function tone(t) {
  if (t < 0) { return C.brick; }
  if (t >= 1) { return C.ink; }
  return "color-mix(in oklab, " + C.ink + " "
    + Math.round(t * 100) + "%, " + C.bg + ")";
}

/* Les séries de la ville, calculées par nyc_live._city() : une part se prend
   sur la somme des courses, jamais sur la moyenne des parts de zones. */
function cityS(key) { return META.city[S.scope][key]; }

function atStep(key, step) {
  const v = cityS(key), n = v.length;
  return v[((step % n) + n) % n];
}

function boot() {
  const g = document.getElementById("grid");
  for (const key of CELLS) {
    g.insertAdjacentHTML("beforeend",
      '<div class="kpi" id="c-' + key + '">'
      + '<div class="k-lab" style="color:' + C.dim + '"></div>'
      + cellBody(key) + "</div>");
  }
  /* Les quatre traits, posés après les cases : ils sont au-dessus du fond et
     sous rien, puisque les cases ne les recouvrent pas. */
  const fade = (dir) => "linear-gradient(to " + dir + ", transparent, "
    + C.rule + " 26%, " + C.rule + " 74%, transparent)";
  for (const k of ["v v1", "v v2", "h h1", "h h2"]) {
    g.insertAdjacentHTML("beforeend", '<i class="sep ' + k
      + '" style="background:' + fade(k[0] === "v" ? "bottom" : "right")
      + '"></i>');
  }
  if (PARTS.pax) { buildWheel(); }
  buildDial();
  buildSeg(document.querySelector("#c-fare .m-seg"), "DD.DD");
  buildSeg(document.querySelector("#c-fare .m-mile"), "D.DD");
  buildSeg(document.querySelector("#c-fare .m-tip"), "DD.D");
  buildNav();
  onState();
}

/* Le corps propre à chaque case, posé entre le titre et le sous-titre. Les
   deux suivent la même disposition — le dessin à gauche, les chiffres à
   droite — pour que les deux rangs se lisent comme une paire et non comme
   deux blocs empruntés ailleurs. */
function cellBody(key) {
  const night = CFG.mode === "light" ? "" : " night";
  if (key === "mph") {
    return '<div class="k-wide"><div class="k-gauge">'
      + '<svg class="k-dial" viewBox="0 0 200 96" aria-hidden="true">'
      + "</svg></div></div>";
  }
  if (key === "fare") {
    return '<div class="k-seg' + night + '"><div class="m-win">'
      + '<div class="m-top"><span class="lb">tarif $</span>'
      + '<svg class="m-seg" viewBox="0 0 ' + segBox("DD.DD")
      + '" aria-hidden="true"></svg></div>'
      /* Deux petits afficheurs sous la course.

         Le prix du mile, parce que c'est lui qui fait le tarif — à 8,21 $ le
         mile et 3,99 mi de trajet, la course vaut 32,76 $. L'étiquette est
         écrite en mots : « $/mi » demandait de deviner qu'il s'agissait d'un
         prix et non d'une distance.

         La part des courses laissant un pourboire, et c'est bien ce que
         l'étiquette dit — « avec pourb. », et non « pourboire », qui aurait
         annoncé un taux. Quatre courses sur cinq, ce qui surprend, et ce
         n'est pourtant qu'un plancher : les 11 à 19 % de courses payées en
         espèces n'enregistrent jamais de pourboire dans les données de la
         TLC, qu'il y en ait eu un ou non. */
      + '<div class="m-row">'
      + '<span class="m-one">'
      + '<svg class="m-sm m-mile" viewBox="0 0 ' + segBox("D.DD")
      + '" aria-hidden="true"></svg>'
      + '<span class="lb">le mile</span></span>'
      + '<span class="m-one">'
      + '<svg class="m-sm m-tip" viewBox="0 0 ' + segBox("DD.D")
      + '" aria-hidden="true"></svg>'
      + '<span class="lb">avec pourb. %</span></span></div>'
      + "</div></div>";
  }
  if (key === "pax") {
    return '<div class="k-wide"><div class="k-wheel">'
      + '<canvas class="k-3d" aria-hidden="true"></canvas>'
      + '<svg class="k-rim" viewBox="0 0 200 100" aria-hidden="true"></svg>'
      + "</div></div>";
  }
  /* Le temps d'abord : c'est ce qu'on lit sur un GPS, et il est en minutes
     et secondes comme une durée de trajet, pas en minutes décimales. */
  return '<div class="k-nav"><canvas class="k-map" aria-hidden="true"></canvas>'
    + navStack() + "</div>";
}

/* Le volant. La jante porte les deux parts, et c'est le seul endroit où les
   données sont : le moyeu et les trois branches sont l'objet, pas la mesure.

   Chaque arc porte sa longueur en dasharray et son départ en dashoffset, donc
   une transition CSS sur ces deux propriétés fait tourner la répartition sans
   la redessiner. L'origine est à midi — `rotate(-90)` — et les arcs tournent
   dans le sens des aiguilles.

   Un jour entre les deux parts, de trois pixels, pris sur la longueur de
   chaque arc : deux aplats qui se touchent se lisent comme un seul. */
/* Le demi-volant. Le moyeu est au bas du cadre et la jante fait un
   demi-tour, de gauche à droite en passant par le haut : les deux parts se
   lisent donc dans l'ordre de la lecture, et non à partir de midi comme sur
   un anneau. Le bas du moyeu sort du cadre, qui le coupe — c'est ce qui en
   fait un volant et non une roue. */
const WH_CX = 100, WH_CY = 96, WH_R = 48, WH_W = 15;
/* Le relief du volant et l'ombre portée changent de force selon le thème :
   un voile blanc sur une page sombre ne fait pas le même effet que sur une
   page claire. */
const NIGHT = CFG.mode === "dark";
/* Les longueurs sont dites en centièmes : `pathLength="100"` fait que le
   navigateur lit les pointillés dans cette échelle, et une part de 24,6 % se
   pose telle quelle. Sans cela il faudrait la longueur de la demi-ellipse,
   que le navigateur et nous ne calculerions pas de la même façon. */
const WH_C = 100;
const WH_GAP = 1.6;                       /* le jour entre deux parts, en % */
/* L'ellipse des étiquettes. Une ellipse parce que la case est deux fois plus
   large que haute ; ses rayons sont pris pour que l'étiquette reste hors de
   la jante et dans le cadre.

   Vérifié sur les trois cycles : « un passager » va de 67,4 % à 90,8 %, donc
   le milieu de son arc tient entre 241 et 262 degrés.

   Trois parts et non deux : « deux passagers » avait disparu dans un
   « deux et plus » qui cachait le plus mobile des trois — la part des courses
   à deux va de 7,4 % à 18,2 % sur la journée, soit deux fois et demie, quand
   celle des solitaires ne bouge que d'un cinquième.

   Mais à trois, les deux petites parts font un arc de 25 et un de 11 degrés,
   et leurs milieux sont à vingt pixels l'un de l'autre : deux étiquettes s'y
   recouvrent. Elles sont donc posées au milieu de leur arc puis écartées
   jusqu'à un pas minimal, en gardant leur ordre — c'est la pose d'étiquettes
   sous contrainte, pas un placement libre. Chacune porte un trait de la
   teinte de sa part, qui dit à quel arc elle appartient quand elle a glissé
   loin de lui. */
/* Le rayon où se posent les étiquettes part de la jante, et s'écarte de la
   demi-largeur ou de la demi-hauteur de l'étiquette selon la direction : une
   boîte posée à rayon constant chevauche la jante dès que son grand côté
   pointe vers elle. C'est la fonction d'appui du rectangle, et c'est ce qui
   garantit qu'aucun chiffre ne passe par-dessus la jante.

   Les deux bouts où elles s'arrêtent : vers l'horizontale, une étiquette
   descendrait sous le cadre de 200 x 100. */
const TAG_PAD = WH_R + WH_W / 2 + 3;
/* Le pas minimal est petit : les étiquettes doivent rester en face de leur
   part, donc l'écartement ne sert qu'à les décoller quand deux parts sont
   voisines et minces. À 0,30 l'étiquette des deux-passagers se retrouvait à
   vingt degrés de son arc ; à 0,13 elle en est à dix, et les deux boîtes ne
   se recouvrent pas — vérifié sur les trois cycles. */
const TAG_MIN = 0.12, TAG_LO = 0.09, TAG_HI = 0.91;

/* Les places des étiquettes sur le demi-tour, en fraction. Une passe vers
   l'avant pour écarter, une vers l'arrière pour rentrer dans le cadre. */
function tagSlots(vals) {
  const n = vals.length, t = [];
  let acc = 0;
  for (let i = 0; i < n; i++) {
    t.push((acc + vals[i] / 2) / 100);
    acc += vals[i];
  }
  t[0] = Math.max(TAG_LO, t[0]);
  for (let i = 1; i < n; i++) { t[i] = Math.max(t[i], t[i - 1] + TAG_MIN); }
  t[n - 1] = Math.min(TAG_HI, t[n - 1]);
  for (let i = n - 2; i >= 0; i--) { t[i] = Math.min(t[i], t[i + 1] - TAG_MIN); }
  return t;
}

/* La jante n'est plus dessinée : elle est rendue. Le SVG ne garde que les
   repères de part, qui doivent rester au-dessus du relief, et les étiquettes
   restent du HTML posé par-dessus.

   Pourquoi un rendu et non un dessin : un tube gainé se lit par ses reflets,
   et un reflet sur un tore est une chose qu'un dégradé SVG approche mais ne
   tient pas — il est juste au milieu de la bande et faux sur les bords, parce
   qu'un dégradé ne sait pas que la surface tourne. Quinze arrêts de dégradé,
   deux filets et un spéculaire flouté donnaient une jante qui se lisait comme
   un ruban peint. La même jante, posée dans une scène avec une lumière et un
   studio en réflexion, se lit comme une pièce. */

/* Les proportions sont celles du SVG, à l'échelle près : rayon 48, bande 15,
   moyeu 21, barre de 46 de demi-largeur sur 13 de haut, dans une tuile de
   200 x 100. Le facteur 1,055 compense l'inclinaison de la caméra : sans lui
   le haut du jonc tombait à 43,5 % de la hauteur au lieu des 40,5 % du SVG. */
const W3 = {
  R: 2.0 * 1.055,
  TILT: 0.30,            /* la jante est couchée vers l'arrière */
  YAW: -0.20,            /* et vue légèrement de côté : sans lacet, un tore
                            vu de face redevient un anneau dessiné */
  YC: -1.8994,           /* le centre tombe à 96 % de la hauteur, comme le SVG */
  FOV: 32, DIST: 7.2,    /* distance sur rayon = 2,95 : c'est la perspective
                            qui donne le volume, pas le nombre de polygones */
  SLOT: 0.152,           /* demi-épaisseur du creux */
  LIT: 0.138,            /* demi-épaisseur de la bande de jauge — la bande
                            élargie. À 0,100 dans un creux de 0,125, elle
                            était un filet au fond d'une gorge : de loin on
                            voyait la gorge, pas la part. Elle remplit
                            maintenant le creux à quatre-vingt-onze pour
                            cent, et c'est la couleur qu'on lit. */
};
W3.TUBE = W3.R * 7.5 / 48;
W3.HUB = W3.R * 21 / 48;
W3.ARM_W = W3.R * 46 / 48;
W3.ARM_H = W3.R * 13 / 48;

/* Un point de la section du jonc. La section est un cercle de rayon TUBE
   centré sur le cercle de rayon R : tout ce qui est à moins de TUBE de ce
   centre est dans la matière, donc invisible. phi = 90 est la face tournée
   vers l'oeil, phi = 0 la lèvre extérieure. */
function w3Tube(phiDeg, push) {
  const a = phiDeg * Math.PI / 180, d = W3.TUBE + (push || 0);
  return { r: W3.R + Math.cos(a) * d, z: Math.sin(a) * d };
}

/* three.js r128 n'a pas de gestion automatique des couleurs : un hexadécimal
   donné à un matériau est pris pour une valeur linéaire. Écrit tel quel avec
   outputEncoding sRGB, le charbon #4e5054 ressortait à #82868c. */
function w3Srgb(hex) {
  return new THREE.Color(hex).convertSRGBToLinear();
}

/* La couleur d'une part passe par un canvas. Le ramp est écrit en
   color-mix(in oklab, ...), que le navigateur rend en color(srgb ...) ou en
   oklab(...) : THREE.Color ne sait lire ni l'un ni l'autre, échoue en silence
   et reste blanc — la part de 75 % sortait blanche dans les deux thèmes. Le
   canvas, lui, sait convertir. */
const W3_SWATCH = document.createElement("canvas");
W3_SWATCH.width = W3_SWATCH.height = 1;
const W3_CTX = W3_SWATCH.getContext("2d");

function w3Css(css) {
  W3_CTX.clearRect(0, 0, 1, 1);
  W3_CTX.fillStyle = "#000000";
  W3_CTX.fillStyle = css;
  W3_CTX.fillRect(0, 0, 1, 1);
  const d = W3_CTX.getImageData(0, 0, 1, 1).data;
  return new THREE.Color(d[0] / 255, d[1] / 255, d[2] / 255)
    .convertSRGBToLinear();
}

/* Le studio en réflexion : un ciel dégradé et trois boîtes lumineuses. Le même
   dans les deux thèmes — c'est lui qui tient la jante au même gris de volant
   sur le panneau clair comme sur le panneau sombre. */
function w3EnvCanvas() {
  const w = 512, h = 256, c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  const sky = g.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, "#f4f7fb");
  sky.addColorStop(0.46, "#c9d4e2");
  sky.addColorStop(0.54, "#8e9aa9");
  sky.addColorStop(1, "#525b67");
  g.fillStyle = sky;
  g.fillRect(0, 0, w, h);
  const box = (cx, cy, rw, rh, a) => {
    const m = Math.max(rw, rh);
    const gr = g.createRadialGradient(cx, cy, 0, cx, cy, m);
    gr.addColorStop(0, "rgba(255,255,255," + a + ")");
    gr.addColorStop(1, "rgba(255,255,255,0)");
    g.save();
    g.translate(cx, cy);
    g.scale(rw / m, rh / m);
    g.translate(-cx, -cy);
    g.fillStyle = gr;
    g.fillRect(cx - rw, cy - rh, rw * 2, rh * 2);
    g.restore();
  };
  box(w * 0.26, h * 0.24, w * 0.20, h * 0.20, 1);
  box(w * 0.74, h * 0.34, w * 0.09, h * 0.22, 0.7);
  box(w * 0.52, h * 0.07, w * 0.30, h * 0.07, 0.5);
  return c;
}

/* L'objet rendu. `set` reçoit les parts en pour cent et les anime. */
const WHEEL = { ok: false, set: () => {} };

function buildWheel3D() {
  const cv = document.querySelector("#c-pax .k-3d");
  if (!cv || typeof THREE === "undefined") { return; }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true,
                                         alpha: true });
  } catch (err) {
    /* Pas de WebGL : la case garde ses étiquettes et ses nombres, et c'est
       la jante seule qui manque. */
    console.warn("volant 3D indisponible", err);
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setClearAlpha(0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(W3.FOV, 2, 0.1, 100);
  camera.position.set(0, 0, W3.DIST);
  camera.lookAt(0, 0, 0);

  const envTex = new THREE.CanvasTexture(w3EnvCanvas());
  envTex.mapping = THREE.EquirectangularReflectionMapping;
  /* Le canvas contient des valeurs sRGB. Sans cette ligne le PMREM les prend
     pour du linéaire et le studio éclaire quatre fois trop. */
  envTex.encoding = THREE.sRGBEncoding;
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  scene.environment = pmrem.fromEquirectangular(envTex).texture;
  envTex.dispose();
  pmrem.dispose();

  const rig = new THREE.Group();
  rig.position.y = W3.YC;
  rig.rotation.x = -W3.TILT;
  rig.rotation.y = W3.YAW;
  scene.add(rig);

  /* Couleur volant, relevée sur une photo de volant de série : le jonc y
     tourne autour de #434444, neutre, avec des reflets jusqu'à #7a7c7e. C'est
     un charbon, pas un gris moyen, et il est le même dans les deux thèmes —
     il se détache du panneau clair comme du panneau sombre. Un jonc charbon
     ne se lit pas par sa matière diffuse, trop sombre pour montrer un
     dégradé : il se lit par ses reflets, d'où un vernis net et serré.

     Pas de grain de cuir : à 200 pixels de large il se moyenne en gris, et
     c'était une texture de 256 carrés par case pour rien. */
  const matLeather = new THREE.MeshPhysicalMaterial({
    color: w3Srgb(0x4e5054), roughness: 0.60, metalness: 0.0,
    clearcoat: 0.50, clearcoatRoughness: 0.28,
    /* Le studio sert les reflets, pas l'éclairage général : à pleine
       intensité il remontait le jonc de #434444 à #82868c. */
    envMapIntensity: 0.45
  });
  const matPlastic = new THREE.MeshStandardMaterial({
    color: w3Srgb(0x3c3d41), roughness: 0.52, metalness: 0.08,
    envMapIntensity: 0.5
  });
  const matPad = new THREE.MeshStandardMaterial({
    color: w3Srgb(0x2e2f33), roughness: 0.78, metalness: 0.04,
    envMapIntensity: 0.5
  });
  const matChrome = new THREE.MeshStandardMaterial({
    color: w3Srgb(0xa8adb4), roughness: 0.26, metalness: 1.0
  });

  rig.add(new THREE.Mesh(new THREE.TorusGeometry(W3.R, W3.TUBE, 24, 180),
                         matLeather));

  /* Les coutures, posées SUR la surface du jonc et non dedans, à cinquante
     degrés de part et d'autre de la face avant. */
  [50, 130].forEach(phi => {
    const p = w3Tube(phi, 0);
    const st = new THREE.Mesh(
      new THREE.TorusGeometry(p.r, 0.013, 8, 160),
      new THREE.MeshStandardMaterial({ color: w3Srgb(0x9a8c6e),
                                       roughness: 0.85 }));
    st.position.z = p.z;
    rig.add(st);
  });

  /* La barre horizontale du SVG, en volume, puis le moyeu et son cache. */
  const arm = new THREE.Mesh(
    new THREE.CylinderGeometry(W3.ARM_H * 0.5, W3.ARM_H * 0.5,
                               W3.ARM_W * 2, 18), matPlastic);
  arm.rotation.z = Math.PI / 2;
  arm.scale.set(1, 1, 0.42);
  arm.position.z = -0.03;
  rig.add(arm);

  const hub = new THREE.Mesh(
    new THREE.CylinderGeometry(W3.HUB * 0.97, W3.HUB, 0.34, 44), matPlastic);
  hub.rotation.x = Math.PI / 2;
  hub.position.z = -0.06;
  rig.add(hub);

  const hubRing = new THREE.Mesh(
    new THREE.TorusGeometry(W3.HUB * 0.97, 0.034, 12, 56), matChrome);
  hubRing.position.z = 0.10;
  rig.add(hubRing);

  const pad = new THREE.Mesh(
    new THREE.CylinderGeometry(W3.HUB * 0.70, W3.HUB * 0.84, 0.11, 44),
    matPad);
  pad.rotation.x = Math.PI / 2;
  pad.position.z = 0.15;
  rig.add(pad);

  const seam = new THREE.Mesh(
    new THREE.BoxGeometry(W3.HUB * 1.2, 0.020, 0.020),
    new THREE.MeshStandardMaterial({ color: w3Srgb(0x2b2b2e),
                                     roughness: 0.95 }));
  seam.position.z = 0.205;
  rig.add(seam);

  [0, 120, 240].forEach(deg => {
    const a = (deg - 90) * Math.PI / 180, rs = W3.HUB * 0.56;
    const screw = new THREE.Mesh(
      new THREE.CylinderGeometry(0.034, 0.034, 0.03, 10),
      new THREE.MeshStandardMaterial({ color: w3Srgb(0x3a3a3d),
                                       roughness: 0.6, metalness: 0.4 }));
    screw.rotation.x = Math.PI / 2;
    screw.position.set(Math.cos(a) * rs, Math.sin(a) * rs, 0.20);
    rig.add(screw);
  });

  const boss = new THREE.Mesh(
    new THREE.CylinderGeometry(W3.HUB * 0.38, W3.HUB * 0.38, 0.07, 28),
    matPlastic);
  boss.rotation.x = Math.PI / 2;
  boss.position.z = 0.23;
  rig.add(boss);
  const bossRing = new THREE.Mesh(
    new THREE.TorusGeometry(W3.HUB * 0.38, 0.018, 8, 36), matChrome);
  bossRing.position.z = 0.255;
  rig.add(bossRing);

  /* La gorge : un creux sombre posé sur la face avant du jonc, dans lequel
     les parts s'allument. C'est le creux qui donne le relief, la bande n'a
     qu'à porter la couleur. */
  const base = w3Tube(90, 0);
  const slot = new THREE.Mesh(
    new THREE.TorusGeometry(base.r, W3.SLOT, 10, 150, Math.PI),
    new THREE.MeshStandardMaterial({ color: w3Srgb(0x15161a), roughness: 0.92,
                                     envMapIntensity: 0.3 }));
  slot.position.z = base.z;
  rig.add(slot);

  /* La face avant du creux avance de SLOT. La bande doit donc avancer d'au
     moins SLOT - LIT pour ne pas disparaître dans le solide du creux : c'est
     exactement ce qui la rendait invisible. Elle dépasse ici de 0,010. */
  const band = w3Tube(90, W3.SLOT - W3.LIT + 0.010);

  /* Un tore par part, refait à sa longueur quand la part change.

     Il était construit une fois sur le demi-tour entier, et chaque part n'en
     montrait qu'un début par setDrawRange. C'était faux : dans un
     TorusGeometry de three.js, les index sont écrits anneau radial par
     anneau radial, et chaque anneau court sur tout l'arc. Tronquer le buffer
     ne raccourcit donc pas l'arc — il rogne la largeur de la bande. Les
     trois parts sortaient à trois largeurs différentes et se recouvraient.

     Refaire la géométrie coûte trois tores de mille sommets par image
     d'animation, soit vingt-cinq images par pas de cycle. C'est le prix d'un
     arc juste, et il est petit. */
  const RAD = 12;
  const arcs = PARTS.pax.map(p => {
    /* toneMapped à faux : la bande doit sortir exactement à la couleur du
       jeton. Le tone-mapping ACES, lui, est fait pour les surfaces éclairées.
       Et un matériau éclairé repeignait la part : la part grise de 75 %
       sortait blanche, et le ramp sur l'encre disparaissait. */
    const mat = new THREE.MeshBasicMaterial({ color: w3Css(partCol(p)),
                                              toneMapped: false });
    const mesh = new THREE.Mesh(
      new THREE.TorusGeometry(band.r, W3.LIT, RAD, 8, 0.001), mat);
    mesh.position.z = band.z;
    rig.add(mesh);
    return { mat: mat, mesh: mesh };
  });

  /* Éclairage contrasté, puisque c'est lui qui sculpte : une clé franche, un
     contre-jour qui détache la silhouette, et peu d'ambiante — une ambiante
     forte remplissait les ombres et réaplatissait la pièce. */
  const keyL = new THREE.DirectionalLight(0xffffff, 1.15);
  keyL.position.set(-4, 5, 6);
  scene.add(keyL);
  const rimL = new THREE.DirectionalLight(0xffffff, 0.8);
  rimL.position.set(5, -1, -4);
  scene.add(rimL);
  scene.add(new THREE.AmbientLight(0xffffff, 0.14));

  let W = 0, H = 0;
  function draw() {
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) { return; }
    if (w !== W || h !== H) {
      W = w; H = h;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    renderer.render(scene, camera);
  }

  /* La pose des parts. Un tore tourne dans le sens inverse des aiguilles
     depuis l'axe des x : on le place par sa fin, jamais par un miroir qui
     retournerait aussi ses normales. */
  const gapF = WH_GAP / WH_C;
  function lay(vals) {
    let acc = 0;
    arcs.forEach((a, i) => {
      const frac = Math.max(vals[i], 0) / WH_C;
      /* Le jour est pris moitié à chaque bout de l'arc, donc le milieu de la
         part ne bouge pas quand le jour change. */
      const len = Math.max(frac - gapF, 0) * Math.PI;
      /* Le pas de segment est pris sur la longueur, pas sur le demi-tour :
         une part de huit pour cent n'a pas besoin de cent quarante segments,
         et une part de soixante-quinze en a besoin de cent cinquante. */
      a.mesh.geometry.dispose();
      a.mesh.geometry = new THREE.TorusGeometry(
        band.r, W3.LIT, RAD, Math.max(6, Math.round(len * 70)),
        Math.max(len, 0.0001));
      a.mesh.visible = len > 0.0005;
      a.mesh.rotation.z = Math.PI - (acc + frac - gapF / 2) * Math.PI;
      acc += frac;
    });
    draw();
  }

  /* L'animation d'un pas : les parts glissent vers leur nouvelle valeur sur
     la durée que le composant pose dans --arc, la même que les nombres et les
     étiquettes. Sans elle, les arcs sautaient pendant que tout le reste
     glissait. */
  const cur = PARTS.pax.map(() => 0);
  let from = cur.slice(), to = cur.slice(), t0 = 0, dur = 0, raf = 0;
  const still = window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function tick(now) {
    const k = dur > 0 ? Math.min((now - t0) / dur, 1) : 1;
    /* La même courbe que les transitions CSS de la case. */
    const e = 1 - Math.pow(1 - k, 3);
    lay(from.map((v, i) => v + (to[i] - v) * e));
    raf = k < 1 ? requestAnimationFrame(tick) : 0;
    if (!raf) { from = to.slice(); }
  }

  WHEEL.ok = true;
  WHEEL.set = vals => {
    to = vals.slice();
    if (still) { from = to.slice(); lay(to); return; }
    const css = getComputedStyle(document.documentElement)
      .getPropertyValue("--arc").trim();
    dur = css.endsWith("ms") ? parseFloat(css)
        : css.endsWith("s") ? parseFloat(css) * 1000 : 420;
    t0 = performance.now();
    if (!raf) { raf = requestAnimationFrame(tick); }
  };

  lay(cur);
  if (window.ResizeObserver) { new ResizeObserver(draw).observe(cv); }
  else { window.addEventListener("resize", draw); }
}

function buildWheel() {
  const el = document.querySelector("#c-pax .k-rim");
  const R1 = WH_R + WH_W / 2;
  buildWheel3D();
  /* Le repère de chaque part, juste en dehors de la bande : un trait de la
     couleur de la part, posé au milieu de son arc, qui tourne quand l'arc
     change de taille. C'est lui qui lie l'étiquette à sa part.

     Sans lui, l'étiquette restait souvent immobile d'un pas à l'autre : la
     règle d'écartement la retient quand deux parts voisines sont minces, et
     une étiquette qui ne bouge pas ne dit pas qu'elle appartient à un arc
     qui, lui, a bougé. Le repère n'a personne à éviter, donc il est toujours
     exactement au milieu de sa part. */
  el.innerHTML = PARTS.pax.map((p, i) =>
    '<g class="pt pt' + i + '" transform="rotate(0 ' + WH_CX + " "
    + WH_CY + ')"><line x1="' + (WH_CX - R1 - 1.5) + '" y1="' + WH_CY
    + '" x2="' + (WH_CX - R1 - 5.5) + '" y2="' + WH_CY + '" stroke="'
    + partInk(p) + '" stroke-width="2" stroke-linecap="round"/></g>')
    .join("");
  buildTags();
}

/* La flèche de position et sa boussole. Le plan de rues l'avait remplacée un
   temps : une trame de rues autour d'une flèche redevenait un écran, et
   c'était l'écran qu'on venait d'enlever. Ici il n'y a que des traits sur un
   cercle, arrondis, et la flèche au milieu.

   Le tout est couché : un facteur sur les ordonnées suffit, et la rotation de
   la flèche se fait dans le groupe couché, donc elle tourne dans le plan du
   sol et non dans celui de l'écran. C'est la même vue que la carte de la
   ville deux blocs plus bas, et c'est la vue qu'on a d'une boussole tenue
   dans la main.

   Le cône est l'incertitude du cap, comme sur un téléphone — large à la
   pointe, éteint au bord. Il tourne avec la flèche parce qu'il est dans le
   même groupe. */
/* ------------------------------------------------------------------
   L'écran de navigation : une maquette de quartier new-yorkais.

   La trame est celle du plan de 1811, à l'échelle, une unité pour trois
   mètres : les rues sont espacées de 200 pieds (61 m, 20 unités), l'avenue
   fait 100 pieds de large (10 unités), la rue transversale 60 (6 unités), et
   la parcelle type 25 pieds de façade (2,5 unités). L'îlot new-yorkais est
   quatre fois et demie plus long que large, et c'est ce rapport-là qui fait
   lire New York plutôt qu'un damier — d'où un seul axe d'avenue dans le
   cadre, et des rues qui le coupent tous les 20.

   Tout est de la même matière : un blanc mat à deux pour cent de clarté
   près pour le sol, la chaussée, les façades et les toits. C'est l'ombre
   qui donne le volume, et rien d'autre. Une chaussée photo-réaliste sous
   des immeubles en maquette, c'étaient deux langages dans une image.

   Rien n'est animé : la géométrie est posée une fois, et seul le ruban
   d'itinéraire change de longueur d'un pas à l'autre.
   ------------------------------------------------------------------ */
const NV3 = {
  PZ: 20,                /* l'entre-deux-rues */
  AVW: 10, STW: 6,       /* largeur de l'avenue, largeur d'une transversale */
  LOT: 2.5,              /* la façade d'une parcelle */
  XMAX: 44,              /* au-delà, rien n'entre dans le cadre */
  /* La caméra est à vingt-six, dans l'axe de l'avenue. Sa pente de visée
     vaut 27,5 degrés et la moitié de son champ vertical 22 : la première
     dépasse la seconde, donc l'horizon sort du cadre et il n'y a pas de
     ciel. La tuile porte alors sa propre teinte dans les deux thèmes. */
  POS: [3, 26, 38], LOOK: [0, 0, -12], FOV: 44,
  /* Le panneau de verre couvre la moitié gauche : la maquette glisse vers
     la part restée visible. */
  DX: -4,
  RX: 2.2, RW: 3.0,      /* le ruban, sur la voie de droite de l'avenue */
  AZ: -3, ARROW: 2.6,    /* le curseur, posé dessus */
  ZA: 34, ZB: -4, ZC: -144  /* le ruban va de ZA à ZB quand la course est
                               la plus courte du cycle, à ZC quand elle est
                               la plus longue */
};
let NAV = null;

/* Le panneau de verre et ses deux lignes. Les icônes sont un ruban
   d'itinéraire et un cadran d'horloge, au trait, à la taille du texte. */
function navStack() {
  const glass = CSS.supports("backdrop-filter", "blur(1px)")
             || CSS.supports("-webkit-backdrop-filter", "blur(1px)");
  /* La maquette est blanche dans les deux thèmes. Un panneau clair posé
     dessus peut donc être très transparent sans perdre son encre sombre ;
     un panneau sombre, non — il vire au gris moyen et son encre claire s'y
     noie. D'où deux valeurs. */
  const bg = glass ? navRgba(C.panel, CFG.mode === "light" ? 0.56 : 0.76)
                   : C.panel;
  const line = navRgba(C.rule, glass ? 0.55 : 1);
  const row = (ico, span, k, top) =>
    '<div class="row" style="' + (top ? "border-top:1px solid " + line : "")
    + '"><svg viewBox="0 0 20 20" stroke="' + C.dim + '">' + ico + "</svg>"
    + '<div><div class="v" style="color:' + C.ink + '">' + span
    + '</div><div class="k" style="color:' + C.dim + '">' + k
    + "</div></div></div>";
  return '<div class="k-stack" style="background:' + bg + ";border:1px solid "
    + line + '">'
    + row('<path d="M6 17c0-4 8-4 8-8a3 3 0 0 0-6 0"/>'
          + '<circle cx="6" cy="17" r="1.6"/>',
          '<span class="n-dist"></span><i>mi</i>', "distance", false)
    + row('<circle cx="10" cy="10" r="7"/><path d="M10 6v4.3l2.8 1.7"/>',
          '<span class="n-dur"></span>', "durée", true)
    + "</div>";
}

/* Une couleur de jeton, rendue transparente. Les jetons sont des #rrggbb :
   le panneau a besoin d'un alpha, et color-mix n'est pas tenu partout dans
   une iframe. */
function navRgba(hex, a) {
  const n = parseInt(hex.replace("#", ""), 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + ","
    + (n & 255) + "," + a + ")";
}

/* Le curseur : une flèche à bords arrondis. Chaque sommet est remplacé par
   une courbe, puis la forme est extrudée avec un biseau — une pointe de
   navigation est une pièce, pas un triangle découpé. */
function navBlade(scale, mat) {
  const pts = [[0, 1.45], [1.1, -1.1], [0, -0.54], [-1.1, -1.1]]
    .map(q => [q[0] * scale, q[1] * scale]);
  const r = 0.24 * scale;
  const sh = new THREE.Shape();
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const a = pts[(i + n - 1) % n], b = pts[i], c = pts[(i + 1) % n];
    const la = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const lc = Math.hypot(c[0] - b[0], c[1] - b[1]);
    const ra = Math.min(r, la / 2), rc = Math.min(r, lc / 2);
    const p1 = [b[0] + (a[0] - b[0]) * ra / la, b[1] + (a[1] - b[1]) * ra / la];
    const p2 = [b[0] + (c[0] - b[0]) * rc / lc, b[1] + (c[1] - b[1]) * rc / lc];
    if (i === 0) { sh.moveTo(p1[0], p1[1]); } else { sh.lineTo(p1[0], p1[1]); }
    sh.quadraticCurveTo(b[0], b[1], p2[0], p2[1]);
  }
  sh.closePath();
  const m = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, {
    depth: 0.3 * scale, bevelEnabled: true, bevelSize: 0.09 * scale,
    bevelThickness: 0.09 * scale, bevelSegments: 3, curveSegments: 16 }), mat);
  m.rotation.x = -Math.PI / 2;
  return m;
}

function buildNav() {
  const cv = document.querySelector("#c-dist .k-map");
  if (!cv || typeof THREE === "undefined") { return; }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true,
                                         alpha: true });
  } catch (e) { return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.96;
  renderer.setClearAlpha(0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(NV3.FOV, 1.8, 0.5, 400);
  cam.position.set(NV3.POS[0] + NV3.DX, NV3.POS[1], NV3.POS[2]);
  cam.lookAt(NV3.LOOK[0] + NV3.DX, NV3.LOOK[1], NV3.LOOK[2]);

  const matGround = new THREE.MeshStandardMaterial({
    color: w3Srgb(0xd9d6d0), roughness: 1.0 });
  const matRoad = new THREE.MeshStandardMaterial({
    color: w3Srgb(0xefedea), roughness: 0.98 });
  const matSide = new THREE.MeshStandardMaterial({
    color: w3Srgb(0xe6e3de), roughness: 0.95 });
  const matRoof = new THREE.MeshStandardMaterial({
    color: w3Srgb(0xf4f2ee), roughness: 0.95 });

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), matGround);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.04;
  ground.receiveShadow = true;
  scene.add(ground);

  const slab = (w, d, x, z) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.1, d), matRoad);
    m.position.set(x, 0.01, z);
    m.receiveShadow = true;
    scene.add(m);
  };
  slab(NV3.AVW, 440, 0, -60);
  for (let k = -8; k <= 2; k++) {
    slab(2 * NV3.XMAX + 30, NV3.STW, 0, k * NV3.PZ);
  }

  /* Générateur à congruence linéaire : la maquette est la même à chaque
     rendu et à chaque rechargement. Une ville qui change de forme quand on
     change de cycle dirait qu'elle porte la donnée, ce qu'elle ne fait pas.

     Chaque bande entre deux rues porte deux rangées dos à dos, et chaque
     rangée avance parcelle par parcelle depuis l'avenue : c'est ce qui donne
     le front bâti continu. Une parcelle sur sept porte une tour, et la tour
     se retire sur sa base, comme l'impose le zonage de 1916. */
  let seed = 20251109;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  const box = (w, h, d, x, y, z) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
      [matSide, matSide, matRoof, matRoof, matSide, matSide]);
    b.position.set(x, y + h / 2, z);
    b.castShadow = true;
    b.receiveShadow = true;
    scene.add(b);
  };
  const ROW = (NV3.PZ - NV3.STW) / 2;
  for (let k = -8; k < 2; k++) {
    const z0 = k * NV3.PZ + NV3.STW / 2;
    for (let i = 0; i < 2; i++) {
      const rz = z0 + ROW * (i + 0.5);
      for (let sgn = -1; sgn <= 1; sgn += 2) {
        let x = NV3.AVW / 2;
        while (x < NV3.XMAX) {
          const w = NV3.LOT * (1 + Math.floor(rnd() * 3));
          const cx = sgn * (x + w / 2);
          const d = ROW - 0.5;
          if (rnd() < 0.14) {
            const h = 9 + rnd() * 5;
            box(w - 0.2, h, d, cx, 0, rz);
            box(w * 0.62, 17 + rnd() * rnd() * 32, d * 0.62, cx, h, rz);
          } else {
            box(w - 0.2, 7 + rnd() * 9.5, d, cx, 0, rz);
          }
          x += w;
        }
      }
    }
  }

  /* L'itinéraire. Sans lui la tuile n'est pas un GPS, c'est une maquette.
     Le ruban est construit sur une longueur de 1 et mis à l'échelle par
     setNav : refaire sa géométrie à chaque pas serait une allocation pour
     un dessin qui ne change que de longueur.

     Pas de virage : vu depuis l'axe de l'avenue, une transversale n'est
     qu'un interstice entre deux fronts bâtis, et tout ce qui s'y passe est
     masqué. */
  const matRoute = new THREE.MeshStandardMaterial({
    color: w3Srgb(0x6f93c4), roughness: 0.5 });
  const ribbon = new THREE.Mesh(
    new THREE.BoxGeometry(NV3.RW, 0.09, 1), matRoute);
  ribbon.position.set(NV3.RX, 0.1, 0);
  scene.add(ribbon);
  const cap = new THREE.Mesh(
    new THREE.CylinderGeometry(NV3.RW / 2, NV3.RW / 2, 0.09, 24), matRoute);
  cap.position.set(NV3.RX, 0.1, 0);
  scene.add(cap);

  /* Le curseur. Il ne se déplace jamais — sur un GPS, c'est la carte qui
     glisse sous la flèche — mais il s'oriente : son cap porte la distance de
     la course, comme le faisait la rose des vents qu'il remplace. C'est la
     lecture nord-en-haut d'un GPS, où la carte est fixe et la flèche tourne.

     Les deux lames sont dans un groupe posé au point du curseur : une
     rotation d'Euler sur la lame elle-même serait appliquée avant son
     couchage, donc elle la ferait pivoter dans son propre plan au lieu de
     la faire tourner à plat. */
  const steer = new THREE.Group();
  steer.position.set(NV3.RX, 0, NV3.AZ);
  scene.add(steer);
  const rim = navBlade(NV3.ARROW * 1.22, new THREE.MeshStandardMaterial({
    color: w3Srgb(0xffffff), roughness: 0.5 }));
  rim.position.y = 0.22 * NV3.ARROW;
  rim.castShadow = true;
  steer.add(rim);
  const head = navBlade(NV3.ARROW, new THREE.MeshStandardMaterial({
    color: w3Srgb(0xe2452c), roughness: 0.4 }));
  head.position.y = 0.46 * NV3.ARROW;
  steer.add(head);

  /* Lumière de maquette : un ciel large et doux, une clé qui pose les
     ombres. Sur du blanc, une clé trop forte brûle les toits et la ville
     perd son relief. */
  const key = new THREE.DirectionalLight(0xffffff, 1.25);
  key.position.set(-70, 110, 60);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = -70;
  key.shadow.camera.right = 70;
  key.shadow.camera.top = 100;
  key.shadow.camera.bottom = -100;
  key.shadow.camera.near = 10;
  key.shadow.camera.far = 300;
  key.shadow.bias = -0.0016;
  scene.add(key);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9a968e, 0.82));

  let W = 0, H = 0;
  const draw = () => {
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) { return; }
    if (w !== W || h !== H) {
      W = w; H = h;
      renderer.setSize(w, h, false);
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
    }
    renderer.render(scene, cam);
  };
  NAV = { draw: draw, ribbon: ribbon, cap: cap, steer: steer };
  draw();
  if (window.ResizeObserver) { new ResizeObserver(draw).observe(cv); }
}

/* Ce que la maquette dit, et elle ne dit que ça : la distance de la course.
   Elle le dit deux fois — par le cap du curseur et par la longueur du ruban
   qui court devant lui. Deux fois la même mesure, donc aucune ne contredit
   l'autre ; le cap se lit d'un coup d'oeil, la longueur se lit en regardant
   où la route s'arrête. Le reste de la ville est du décor et ne prétend
   rien mesurer.

   La course du cap est de 70 degrés, 35 de part et d'autre de l'axe de
   l'avenue. Au-delà, la flèche pointe dans une façade. */
function setNav(cell, step, delay) {
  rollCell(cell.querySelector(".n-dur"), "dist/min",
           atStep("dur", step), mmss, delay);
  rollCell(cell.querySelector(".n-dist"), "dist/mi",
           atStep("dist", step), x => un(x, 2), delay);
  if (!NAV) { return; }
  const f = diFrac(atStep("dist", step), cycleRange("dist"));
  const zEnd = NV3.ZB + (NV3.ZC - NV3.ZB) * f;
  const len = NV3.ZA - zEnd;
  NAV.ribbon.scale.z = len;
  NAV.ribbon.position.z = (NV3.ZA + zEnd) / 2;
  NAV.cap.position.z = zEnd;
  /* Positif vers la gauche : une rotation autour de l'axe vertical emmène
     la pointe, qui regarde vers les z négatifs, vers les x négatifs. La
     course courte part à gauche, la longue à droite, comme le faisait le
     cap de la rose. */
  NAV.steer.rotation.y = (0.5 - f) * 70 * Math.PI / 180;
  NAV.draw();
}

/* Les deux étiquettes, posées par-dessus le volant. Elles sont dans le bloc
   et non dans le SVG : leur place suit la part, et une transition de
   `transform` sur un élément HTML est tenue partout. */
function buildTags() {
  const el = document.querySelector("#c-pax .k-wheel");
  el.insertAdjacentHTML("beforeend", PARTS.pax.map((p, i) =>
    '<div class="tag" id="tg-pax-' + i + '" style="color:' + C.ink + '">'
    /* Le dessin porte la couleur de sa part : c'est ce qui dit de quel arc
       l'étiquette parle, maintenant que le trait de teinte est parti. Le
       chiffre, lui, reste à l'encre du texte. */
    + '<span class="kt" style="color:' + partInk(p) + '">' + p.fig + "</span>"
    + '<b class="kv" id="kv-pax-' + i + '"></b></div>').join(""));
}

/* Le cadran. Course de DI_SWEEP degrés à partir de DI_A0, mesuré depuis
   l'est et dans le sens des aiguilles — l'axe des ordonnées de SVG
   descendant, c'est bien le sens horaire à l'écran. À f = 0 l'aiguille
   pointe en bas à gauche, à f = 0,5 vers le haut, à f = 1 en bas à droite :
   la course d'un compteur de voiture.

   L'aiguille est dessinée une fois dans la direction de f = 0, puis
   tournée. Un sinus par trame donnerait le même résultat pour plus de
   calcul, et surtout sans transition CSS. */
const DI_CX = 100, DI_CY = 56, DI_R = 40, DI_A0 = 150, DI_SWEEP = 240;

function diPt(f, r) {
  const a = (DI_A0 + f * DI_SWEEP) * Math.PI / 180;
  return [DI_CX + r * Math.cos(a), DI_CY + r * Math.sin(a)];
}

/* Un trait de graduation. L'épaisseur et l'opacité sont données : la
   graduation a trois rangs, et c'est leur inégalité qui la fait lire. */
function diTick(f, r1, r2, w, col, op) {
  const a = diPt(f, r1), b = diPt(f, r2);
  return '<line x1="' + a[0].toFixed(2) + '" y1="' + a[1].toFixed(2)
    + '" x2="' + b[0].toFixed(2) + '" y2="' + b[1].toFixed(2) + '" stroke="'
    + col + '" stroke-width="' + w + '" stroke-linecap="round" opacity="'
    + op + '"/>';
}

/* Une graduation principale : un coin, large au pied et étroit à la pointe,
   et non un trait épais à bouts ronds. C'est la taille des graduations d'un
   cadran de sport — la pièce est emboutie, donc elle a une arête. */
function diWedge(f, r1, r2, w1, w2, col) {
  const a = (DI_A0 + f * DI_SWEEP) * Math.PI / 180;
  const cs = Math.cos(a), sn = Math.sin(a);
  const px = -sn, py = cs;
  const pt = (r, w) => (DI_CX + cs * r + px * w).toFixed(2) + ","
    + (DI_CY + sn * r + py * w).toFixed(2);
  return '<polygon class="maj" points="' + pt(r1, w2) + " " + pt(r2, w1)
    + " " + pt(r2, -w1) + " " + pt(r1, -w2) + '" fill="' + col + '"/>';
}

/* ------------------------------------------------------------------
   L'écran à cristaux liquides du cadran.

   Les couleurs ne sont pas choisies, elles sont relevées sur la photo d'un
   afficheur réel, pixel par pixel : le verre va de #86a94e en haut à gauche
   à #637f3f en bas à droite, et l'encre des segments est #1f3413, un olive
   très sombre et jamais un noir.

   Ce qui fait lire un écran plutôt qu'un rectangle vert, dans l'ordre : les
   segments éteints, dessinés sous les chiffres ; le creux, une paroi sombre
   et l'ombre que son bord haut porte sur le verre ; et les chiffres à sept
   segments, penchés de sept degrés comme sur un afficheur de série.
   ------------------------------------------------------------------ */

/* Un barreau horizontal et un barreau vertical, tous deux en hexagone :
   c'est la forme d'un segment réel, pointu aux deux bouts pour que deux
   segments voisins se rejoignent sans se chevaucher. */
function diHbar(x, y, L, t) {
  const h = t / 2;
  return [[x, y], [x + h, y - h], [x + L - h, y - h], [x + L, y],
          [x + L - h, y + h], [x + h, y + h]];
}
function diVbar(x, y, L, t) {
  const h = t / 2;
  return [[x, y], [x + h, y + h], [x + h, y + L - h], [x, y + L],
          [x - h, y + L - h], [x - h, y + h]];
}
function diPoly(pts, attr) {
  return '<polygon points="' + pts.map(q => q[0].toFixed(2) + ","
    + q[1].toFixed(2)).join(" ") + '" ' + attr + "/>";
}

const DI_SEG7 = { "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd",
                  "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
                  "8": "abcdefg", "9": "abcdfg" };

/* Un chiffre. x et y sont le coin haut gauche de la cellule, w et h sa
   taille, t l'épaisseur du segment, g le jour laissé aux angles. Les
   segments éteints sont toujours dessinés : c'est eux qu'on voit sur un
   vrai écran, et c'est à eux qu'on le reconnaît. */
function diDigit(ch, x, y, w, h, t, g) {
  const on = DI_SEG7[ch] || "";
  const half = h / 2;
  const bars = {
    a: diHbar(x + g, y, w - 2 * g, t),
    g: diHbar(x + g, y + half, w - 2 * g, t),
    d: diHbar(x + g, y + h, w - 2 * g, t),
    f: diVbar(x, y + g, half - 2 * g, t),
    b: diVbar(x + w, y + g, half - 2 * g, t),
    e: diVbar(x, y + half + g, half - 2 * g, t),
    c: diVbar(x + w, y + half + g, half - 2 * g, t)
  };
  let out = "";
  for (const k of Object.keys(bars)) {
    out += diPoly(bars[k],
      'class="' + (on.indexOf(k) < 0 ? "seg-off" : "seg-on") + '"');
  }
  return out;
}

/* L'écran entier. w et h sont ceux du verre ; le creux et la paroi sont
   dessinés autour. Le cadre ne dépasse jamais le dernier nombre du cadran :
   à 34 de large centré sur l'axe, son bord droit tombe à 117, et le nombre
   de droite commence à 118. */
function diLcd(w, h, y, val, ann) {
  const x = DI_CX - w / 2;
  const rect = (dx, dy, rx, extra) =>
    '<rect x="' + (x - dx).toFixed(2) + '" y="' + (y - dy).toFixed(2)
    + '" width="' + (w + dx * 2).toFixed(2) + '" height="'
    + (h + dy * 2).toFixed(2) + '" rx="' + rx + '" ' + extra + "/>";
  /* Les chiffres sont calés à droite. Une cellule fait cw de large, la
     virgule un bloc carré, et l'ensemble est penché de sept degrés. */
  const cw = 4.4, ch = 8.2, ct = 1.05, gp = 0.32, sp = 1.9, cm = 2.2;
  const chars = String(val).split("");
  let wide = 0;
  for (const c of chars) { wide += (c === "," ? cm : cw + sp); }
  let cx0 = x + w - 3.0 - wide;
  /* Les chiffres occupent le bas de la vitre : la rangée du haut est
     réservée à l'annonciateur, comme sur une calculatrice scientifique où
     le DEG et le RAD sont au-dessus du nombre et jamais à côté. */
  const cy0 = y + h - 1.8 - ch;
  let digits = "";
  for (const c of chars) {
    if (c === ",") {
      digits += diPoly([[cx0, cy0 + ch], [cx0 + ct, cy0 + ch],
                        [cx0 + ct, cy0 + ch + ct], [cx0, cy0 + ch + ct]],
                       'class="seg-on"');
      cx0 += cm;
    } else {
      digits += diDigit(c, cx0, cy0, cw, ch, ct, gp);
      cx0 += cw + sp;
    }
  }
  return rect(1.5, 1.5, 3, 'fill="#0c0e0f"')
    + rect(0.7, 0.7, 2.2, 'fill="#2b2f32"')
    + rect(0, 0, 1.6, 'fill="url(#di-glass)"')
    /* L'ombre que le bord haut du creux porte sur le verre. C'est elle qui
       donne le millimètre de profondeur : sans elle, le verre est collé au
       cadran. */
    + rect(0, 0, 1.6, 'fill="url(#di-drop)"')
    + '<text x="' + (x + 3).toFixed(1) + '" y="' + (y + 4.1).toFixed(1)
    + '" class="lcd-ann">' + ann + "</text>"
    + '<g transform="translate(' + (x + w / 2).toFixed(2) + " "
    + (y + h / 2).toFixed(2) + ') skewX(-7) translate('
    + (-(x + w / 2)).toFixed(2) + " " + (-(y + h / 2)).toFixed(2) + ')">'
    + digits + "</g>"
    + rect(0, 0, 1.6, 'fill="url(#di-sheen)"')
    + rect(0, 0, 1.6, 'fill="none" stroke="#f2f6ea" stroke-opacity="0.28"'
           + ' stroke-width="0.5"');
}

/* Les dégradés du cadran. Les identifiants sont préfixés : les quatre cases
   partagent un seul document, et deux defs de même nom s'y écrasent. */
function diDefs() {
  return "<defs>"
    + '<linearGradient id="di-ndlU" x1="0" y1="0" x2="0" y2="1">'
    + '<stop offset="0" stop-color="#ff9f8a"/>'
    + '<stop offset="1" stop-color="#e2452c"/></linearGradient>'
    + '<linearGradient id="di-ndlL" x1="0" y1="0" x2="0" y2="1">'
    + '<stop offset="0" stop-color="#c22d19"/>'
    + '<stop offset="1" stop-color="#7d170a"/></linearGradient>'
    + '<radialGradient id="di-hub" cx="0.33" cy="0.28" r="0.9">'
    + '<stop offset="0" stop-color="#ffffff"/>'
    + '<stop offset="0.42" stop-color="#bcc2ca"/>'
    + '<stop offset="1" stop-color="#5a6068"/></radialGradient>'
    + '<linearGradient id="di-glass" x1="0.1" y1="0" x2="0.9" y2="1">'
    + '<stop offset="0" stop-color="#86a94e"/>'
    + '<stop offset="0.42" stop-color="#749143"/>'
    + '<stop offset="1" stop-color="#637f3f"/></linearGradient>'
    + '<linearGradient id="di-drop" x1="0" y1="0" x2="0" y2="1">'
    + '<stop offset="0" stop-color="#000000" stop-opacity="0.42"/>'
    + '<stop offset="0.3" stop-color="#000000" stop-opacity="0.06"/>'
    + '<stop offset="0.86" stop-color="#000000" stop-opacity="0"/>'
    + '<stop offset="1" stop-color="#ffffff" stop-opacity="0.18"/>'
    + "</linearGradient>"
    + '<linearGradient id="di-sheen" x1="0" y1="0" x2="0.75" y2="1">'
    + '<stop offset="0" stop-color="#ffffff" stop-opacity="0.22"/>'
    + '<stop offset="0.36" stop-color="#ffffff" stop-opacity="0.05"/>'
    + '<stop offset="0.37" stop-color="#ffffff" stop-opacity="0"/>'
    + "</linearGradient>"
    + '<filter id="di-sh" x="-70%" y="-70%" width="240%" height="240%">'
    + '<feDropShadow dx="0.9" dy="1.6" stdDeviation="1"'
    + ' flood-color="#000000" flood-opacity="0.4"/></filter>'
    + "</defs>";
}

/* L'aiguille en volume. Elle est dessinée le long de +x puis tournée de
   DI_A0 : son axe de largeur est alors l'axe des y, et un dégradé vertical
   en coordonnées de boîte lui donne un flanc clair et un flanc sombre.
   Dessinée en diagonale, le même dégradé l'aurait coupée en biais.

   Deux moitiés et non un polygone : c'est l'arête entre les deux qui fait
   le volume. Le contrepoids derrière l'axe la fait lire comme une pièce
   montée sur un pivot, et non comme un rayon. */
function diNeedle(len, hw, tail) {
  const q = (x, y) => x.toFixed(2) + "," + y.toFixed(2);
  const up = [q(len, 0), q(7, -hw), q(-tail, -hw * 0.62), q(-tail, 0),
              q(7, 0)].join(" ");
  const lo = [q(len, 0), q(7, 0), q(-tail, 0), q(-tail, hw * 0.62),
              q(7, hw)].join(" ");
  return '<g transform="translate(' + DI_CX + " " + DI_CY + ") rotate("
    + DI_A0 + ')" filter="url(#di-sh)">'
    + '<polygon points="' + up + '" fill="url(#di-ndlU)"/>'
    + '<polygon points="' + lo + '" fill="url(#di-ndlL)"/></g>';
}

/* Le chapeau d'axe : un dôme chromé, son ombre portée, et la pastille
   rouge de l'aiguille au centre. */
function diHub(r, cap) {
  return '<circle cx="' + DI_CX + '" cy="' + DI_CY + '" r="' + r
    + '" fill="url(#di-hub)" filter="url(#di-sh)"/>'
    + '<circle cx="' + DI_CX + '" cy="' + DI_CY + '" r="' + r
    + '" fill="none" stroke="#000000" stroke-opacity="0.38"'
    + ' stroke-width="0.7"/>'
    + '<circle cx="' + DI_CX + '" cy="' + DI_CY + '" r="'
    + (r * 0.34).toFixed(2) + '" fill="' + cap + '"/>';
}

/* Le cadran est nu : ni disque, ni verre, ni cercle de couronne. Un cercle
   autour de la graduation refermait le dessin sur lui-même ; sans lui, la
   graduation est le cadran. Restent le liseré rouge, la graduation, les
   nombres, l'aiguille, le moyeu et l'écran. */
function buildDial() {
  const el = document.querySelector("#c-mph .k-dial");
  el.innerHTML = diDefs()
    /* Le liseré rouge, sur le dernier cinquième de la course : la zone rouge
       d'un compteur de sport, une bande franche au bord du cadran et non une
       teinture de la course. Il a ici un sens — au-delà, la ville roule plus
       vite qu'à aucun pas du cycle. */
    + '<path d="' + diArcSeg(0.8, 1, 42.3) + '" fill="none" stroke="'
    + C.brick + '" stroke-width="2.2" stroke-linecap="round"/>'
    /* Graduation et nombres sont posés par setDial : ils tombent sur les
       nombres ronds, donc ils dépendent de la borne du cycle. */
    + '<g class="grad"></g>'
    + '<g class="ndl">' + diNeedle(33.5, 2.1, 10) + "</g>"
    + diHub(4.6, "#b02f1b")
    + '<g class="lcd"></g>';
}

/* Un morceau d'arc de la course, entre deux fractions. */
function diArcSeg(f0, f1, r) {
  const a = diPt(f0, r), b = diPt(f1, r);
  const big = (f1 - f0) * DI_SWEEP > 180 ? 1 : 0;
  return "M " + a[0].toFixed(2) + " " + a[1].toFixed(2)
    + " A " + r + " " + r + " 0 " + big + " 1 "
    + b[0].toFixed(2) + " " + b[1].toFixed(2);
}

/* La graduation. Un trait tous les demi-miles, un trait moyen au mile, un
   coin large tous les cinq. Trois rangs et non deux : à deux rangs la
   couronne est une alternance, à trois elle a une respiration, et c'est à
   ça qu'on reconnaît la graduation d'un compteur.

   Le coin est large au bord et fin vers le centre — une graduation est
   emboutie, donc elle a une arête, et non un trait épais à bouts ronds.

   C'est pour cela que la borne haute est arrondie au multiple de cinq
   au-dessus du maximum du cycle — 21,0 mph donne un cadran de 0 à 25. Un
   compteur dont l'aiguille butait pile sur la dernière graduation n'existe
   pas, et les nombres ronds ne tomberaient nulle part.

   Le demi-mile n'est tracé que jusqu'à 30 : au-delà, cinquante traits de
   plus ne sont plus une graduation mais un grisé. */
function diGrad(hi) {
  const step = hi <= 30 ? 0.5 : 1;
  const r = (x) => Math.abs(x - Math.round(x)) < 1e-6;
  let out = "";
  for (let v = 0; v <= hi + 1e-6; v += step) {
    const f = v / hi;
    if (r(v / 5)) { out += diWedge(f, 41, 35, 0.5, 0.85, C.ink); }
    else if (r(v)) { out += diTick(f, 41, 37.4, 0.8, C.dim, 1); }
    else { out += diTick(f, 41, 39.2, 0.55, C.dim, 0.5); }
  }
  /* Les nombres tiennent leur propre couronne, à 28 du centre. Posés au
     pied des graduations ils se gênaient : il fallait les faire gros pour
     qu'ils se détachent, et gros ils mangeaient la graduation. */
  for (let v = 0; v <= hi + 1e-6; v += 5) {
    const q = diPt(v / hi, 28);
    out += '<text class="bound" x="' + q[0].toFixed(1) + '" y="'
      + (q[1] + 3.5).toFixed(1) + '" text-anchor="middle" fill="' + C.ink
      + '">' + v + "</text>";
  }
  return out;
}

/* Les bornes de l'arc et la moyenne du cycle affiché. Mises en cache par
   cycle, la série ne changeant qu'avec lui.

   La course part de zéro et monte au maximum du cycle. Elle allait du
   minimum au maximum, ce qui dilatait l'amplitude mais mentait sur l'objet :
   un compteur de voiture commence à zéro, et une aiguille à mi-course doit
   dire la moitié de la vitesse maximale, pas un rang dans une plage. Le
   maximum, lui, reste celui du cycle affiché : un cadran gradué jusqu'à 30
   aurait laissé l'aiguille dans le tiers bas en permanence sur l'année, qui
   ne va que de 10,6 à 12,8 mph.

   Le repère de la moyenne reste, et c'est lui qui porte maintenant la
   comparaison : l'aiguille dit la vitesse, le repère dit où elle tombe. */
const RANGE = {};
function cycleRange(key) {
  const ck = S.scope + "/" + key;
  if (!RANGE[ck]) {
    const v = cellSeries(key);
    let sum = 0;
    for (const x of v) { sum += x; }
    RANGE[ck] = { lo: 0, hi: Math.max.apply(null, v),
                  avg: v.length ? sum / v.length : 0 };
  }
  return RANGE[ck];
}

/* Le tarif d'une course. Il n'a pas de série à lui dans le payload, et n'en
   a pas besoin : le dollar par mile fois le mile par course donne le dollar
   par course. Les deux séries sont déjà servies, et le produit retombe au
   centime sur la mesure faite en SQL — 24,33 $ à 2 h, 34,76 $ à 5 h.

   C'est un rapport de deux quantités du même pas, donc la normalisation par
   heure s'y annule : le tarif moyen est vrai sur les trois cycles. */
function fareAt(step) {
  return atStep("mile", step) * atStep("dist", step);
}

/* Où tombe une valeur sur la course du cadran, bornée à [0, 1]. */
function diFrac(v, r) {
  return r.hi > r.lo ? Math.max(0, Math.min(1, (v - r.lo) / (r.hi - r.lo))) : 0;
}

/* Pose l'aiguille, le remplissage, le repère de moyenne, les deux bornes,
   la vitesse en chiffres et le compteur kilométrique.

   Les trois nombres de la case se vérifient l'un par l'autre : le cadran
   porte la vitesse, la fenêtre porte la distance, le sous-titre porte la
   durée, et la vitesse est le quotient des deux autres. Le payload le
   garantit — mph = dist / (dur / 60) — donc la case ne peut pas se
   contredire. */
const DI_SHOWN = {};
function setDial(cell, step) {
  const el = cell.querySelector(".k-dial");
  const r0 = cycleRange("mph");
  /* La borne du cadran est le multiple de cinq au-dessus du maximum du
     cycle : c'est ce qui fait tomber les graduations sur des nombres ronds. */
  const hi = Math.max(5, Math.ceil(r0.hi / 5) * 5);
  if (DI_SHOWN.hi !== hi) {
    DI_SHOWN.hi = hi;
    el.querySelector(".grad").innerHTML = diGrad(hi);
  }
  const v = cellValue("mph", step);
  const f = diFrac(v, { lo: 0, hi: hi });
  el.querySelector(".ndl").setAttribute("transform",
    "rotate(" + (f * DI_SWEEP).toFixed(2) + " " + DI_CX + " " + DI_CY + ")");
  /* L'écran ne roule pas ses chiffres : un afficheur à cristaux liquides
     saute d'une valeur à l'autre, il ne les fait pas défiler. L'aiguille,
     elle, glisse — et c'est le partage des rôles d'un vrai compteur, où le
     mouvement est mécanique et le chiffre électronique. */
  el.querySelector(".lcd").innerHTML =
    diLcd(34, 15, DI_CY + 14, un(v, 1), "MPH");
}

/* ------------------------------------------------------------------
   Les chiffres à sept segments du taximètre.

   Un afficheur à diodes se reconnaît à ses segments éteints : sur un
   radio-réveil on voit le huit derrière le un. Une police à pas fixe mise en
   rouge n'en dit rien, et il n'existe pas ici de police à sept segments sous
   licence libre. Les segments sont donc dessinés, les sept par chiffre, et le
   compteur n'allume que ceux de sa valeur.

       a            Chaque segment est un hexagone aux bouts biseautés : c'est
     f   b          la forme d'un segment réel, et c'est ce qui laisse les
       g            petites encoches aux angles.
     e   c
       d
   ------------------------------------------------------------------ */
/* Les segments étaient maigres et écartés : quatre pixels d'épaisseur dans
   une boîte de 20 sur 36, et les montants posés en dehors de la portée des
   barres. Un chiffre à sept segments ne se lit que si ses barres se touchent
   presque aux angles et si leur épaisseur fait environ le sixième de la
   hauteur — sinon ce sont sept traits, et pas un chiffre.

   La boîte fait maintenant 18 sur 32, les barres 4,4 d'épaisseur, et les
   montants partent exactement du bout des barres, donc les angles se
   rejoignent à l'encoche près. À 5,2 les barres étaient grasses et les
   contre-formes du huit se fermaient. */
const SEG_T = 2.2;               /* demi-épaisseur d'un segment */
const SEG_MAP = { "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
                  "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
                  "8": "abcdefg", "9": "abcdfg" };
/* La boîte d'un chiffre fait 18 sur 32. Les trois barres horizontales sont à
   2,8, 16 et 29,2, les quatre montants à 2,8 et 15,2. */
const SEG_GEO = {
  a: ["h", 2.8, 2.8, 15.2], g: ["h", 2.8, 16, 15.2],
  d: ["h", 2.8, 29.2, 15.2],
  f: ["v", 2.8, 2.8, 16], b: ["v", 15.2, 2.8, 16],
  e: ["v", 2.8, 16, 29.2], c: ["v", 15.2, 16, 29.2],
};

function segPoly(g) {
  const t = SEG_T;
  let pts;
  if (g[0] === "h") {
    const x0 = g[1], y = g[2], x1 = g[3];
    pts = [[x0, y], [x0 + t, y - t], [x1 - t, y - t], [x1, y],
           [x1 - t, y + t], [x0 + t, y + t]];
  } else {
    const x = g[1], y0 = g[2], y1 = g[3];
    pts = [[x, y0], [x + t, y0 + t], [x + t, y1 - t], [x, y1],
           [x - t, y1 - t], [x - t, y0 + t]];
  }
  return pts.map(q => q[0] + "," + q[1]).join(" ");
}

/* La mise en page d'un afficheur se donne par un gabarit — « DD.DD », « D.DD »
   — et non par une liste d'abscisses. Les trois afficheurs de la case ont des
   gabarits différents, et une liste par afficheur aurait été trois fois la
   même arithmétique écrite à la main.

   Un chiffre occupe 19 de large, le point 7, et il reste 5 de marge pour le
   cisaillement. Les bornes des trois séries tiennent dans ces gabarits sur
   les trois cycles : la course de 22,93 à 36,70 $, le mile de 5,83 à 9,37 $.

   La part des courses avec pourboire va de 67,5 à 84,8 % et tient donc dans
   « DD.D ». Ce que la série compte exactement : les courses payées par carte
   avec un pourboire enregistré, rapportées à toutes les courses payées —
   carte et espèces. C'est un plancher, puisque les espèces n'enregistrent
   jamais rien, et l'étiquette dit « avec pourb. » et non « pourboire » pour
   ne pas annoncer un taux. */
const SEG_W = 19, SEG_PT = 7, SEG_PAD = 3;

function segLayout(mask) {
  const cells = [], pts = [];
  let x = 0;
  for (const ch of mask) {
    if (ch === ".") { pts.push(x); x += SEG_PT; }
    else { cells.push(x); x += SEG_W; }
  }
  return { cells: cells, pts: pts, w: x + SEG_PAD + 2 };
}

function segBox(mask) { return segLayout(mask).w + " 32"; }

function buildSeg(el, mask) {
  /* Les chiffres sont penchés. C'est la marque d'un afficheur à diodes — les
     réveils, les autoradios, les taximètres le sont tous — et six degrés
     suffisent à le dire. Le cisaillement est pris autour du milieu de la
     hauteur, sinon le bas du nombre glisserait seul. */
  const L = segLayout(mask);
  let html = '<g transform="translate(' + SEG_PAD
    + ' 16) skewX(-6) translate(0 -16)">';
  for (const ox of L.cells) {
    html += '<g class="dig" transform="translate(' + ox + ' 0)">';
    for (const k of "abcdefg") {
      html += '<polygon class="off" data-seg="' + k + '" points="'
        + segPoly(SEG_GEO[k]) + '"/>';
    }
    html += "</g>";
  }
  /* Le point décimal : un carré sur la ligne de base. Un taximètre affiche
     un point et non une virgule, quelle que soit la langue du tableau de
     bord — c'est un appareil, pas une page. */
  for (const ox of L.pts) {
    html += '<g class="dig pt"><polygon class="lit" data-seg="." points="'
      + (ox + 1.1) + ",27 " + (ox + 5.9) + ",27 " + (ox + 5.9) + ",31.8 "
      + (ox + 1.1) + ',31.8"/></g>';
  }
  el.innerHTML = html + "</g>";
}

/* Allume les segments de la valeur. Les séparateurs du texte sont sautés :
   ce qui reste est la suite des chiffres, dans l'ordre des cases. Un chiffre
   absent est une espace, et n'allume rien — comme sur un vrai afficheur, où
   le zéro de tête ne s'affiche pas. */
function paintSeg(el, txt) {
  const chars = [];
  for (const ch of txt) { if (ch !== "." && ch !== ",") { chars.push(ch); } }
  const gs = el.querySelectorAll("g.dig:not(.pt)");
  for (let i = 0; i < gs.length; i++) {
    const on = SEG_MAP[chars[i]] || "";
    const ps = gs[i].querySelectorAll("polygon");
    for (let j = 0; j < ps.length; j++) {
      ps[j].setAttribute("class",
        on.indexOf(ps[j].getAttribute("data-seg")) >= 0 ? "lit" : "off");
    }
  }
}

/* Le nombre, calé à droite sur la longueur de son gabarit : le point reste
   à sa place quand le tarif passe sous dix dollars. L'afficheur étant un
   appareil, le séparateur est un point et non la virgule du texte français. */
function segFmt(x, dec, len) {
  const t = un(x, dec).replace(",", ".");
  return t.length < len ? " ".repeat(len - t.length) + t : t;
}

/* Le taximètre. Ses couleurs sont dans la feuille de style et non posées
   ici — l'afficheur est un objet, pas une case du thème. */
function setMeter(cell, step, delay) {
  rollCell(cell.querySelector(".m-seg"), "fare", fareAt(step),
           v => segFmt(v, 2, 5), delay, paintSeg);
  rollCell(cell.querySelector(".m-mile"), "mile", atStep("mile", step),
           v => segFmt(v, 2, 4), delay + 60, paintSeg);
  rollCell(cell.querySelector(".m-tip"), "tip", atStep("tip", step),
           v => segFmt(v, 1, 4), delay + 120, paintSeg);
}

/* La distance de la course, son temps, et le cap de la flèche. */
/* Une durée en minutes décimales, dite en minutes et secondes. */
function mmss(v) {
  const t = Math.max(0, Math.round(v * 60));
  const m = Math.floor(t / 60), q = t % 60;
  return (m < 10 ? "0" : "") + m + ":" + (q < 10 ? "0" : "") + q;
}

/* Pose les parts : les arcs de la jante, la place des étiquettes, les
   pourcentages. */
function setParts(key, step, delay) {
  const vals = PARTS[key].map(p => partVal(p, step));
  const slots = tagSlots(vals);
  if (key === "pax") {
    const el = document.querySelector("#c-pax .k-rim");
    const wheel = document.querySelector("#c-pax .k-wheel");
    /* Le jour est pris moitié à chaque bout de l'arc, donc le milieu de
       l'arc ne bouge pas et l'étiquette reste centrée sur sa part. */
    WHEEL.set(vals);
    let acc = 0;
    PARTS.pax.forEach((p, i) => {
      /* La place de l'étiquette, écartée de ses voisines, et son rayon pris
         sur sa propre taille : le coin d'une boîte large passerait sur la
         jante à rayon constant. Les tailles de secours servent hors
         navigateur, où rien n'est mesuré. */
      /* Le repère va au milieu exact de la part, mesuré sur les valeurs et
         non sur les places écartées : c'est la donnée, il n'a rien à
         éviter. */
      const pt = el.querySelector(".pt" + i);
      const half = (acc + vals[i] / 2) / WH_C * 180;
      pt.setAttribute("transform",
        "rotate(" + half.toFixed(2) + " " + WH_CX + " " + WH_CY + ")");
      const mid = Math.PI + slots[i] * Math.PI;
      const tg = wheel.querySelector("#tg-pax-" + i);
      const hw = (tg.offsetWidth || 34) / 2, hh = (tg.offsetHeight || 24) / 2;
      const cs = Math.cos(mid), sn = Math.sin(mid);
      const rr = TAG_PAD + hw * Math.abs(cs) + hh * Math.abs(sn);
      const x = WH_CX + rr * cs;
      const y = WH_CY + rr * sn;
      tg.style.transform =
        "translate(calc(" + x.toFixed(1) + "px - 50%), calc("
        + y.toFixed(1) + "px - 50%))";
      acc += vals[i];
    });
  }
  PARTS[key].forEach((p, i) => {
    rollCell(document.getElementById("kv-" + key + "-" + i),
             key + "/" + i, vals[i], x => un(x, 1) + " %",
             delay + i * 50);
  });
}

function un(x, n) { return x.toFixed(n).replace(".", ","); }

function cellValue(key, step) {
  return atStep(key, step);
}

function cellLabel(key) {
  return CELL_LABEL[key];
}

/* La base de la répartition des passagers — la part des courses qui
   déclarent le champ — est dite par le sous-titre de sa case, dans PAINT.
   Un troisième arc « ? » aurait été plus visible, mais il aurait changé la
   question : la jante répondrait « que sait-on des passagers » au lieu de
   « combien de passagers », et à 4 h, où 40 % des courses ne déclarent rien,
   cet arc aurait écrasé les deux autres. La base bouge autant que ce qu'elle
   qualifie, de 60 % à 90 % des courses selon l'heure. */

/* La série de la case sur le cycle entier, mise en cache : elle ne change
   qu'avec le cycle, les deux cases lisant META.city et non le binaire. */
const SPARK = {};
function cellSeries(key) {
  const ck = S.scope + "/" + key;
  if (!SPARK[ck]) {
    const n = stepsOf(S.scope);
    const out = new Array(n);
    for (let i = 0; i < n; i++) { out[i] = cellValue(key, i); }
    SPARK[ck] = out;
  }
  return SPARK[ck];
}


/* Plus de lecture du binaire ici. Les deux cases lisent META.city, qui porte
   déjà les séries de la ville calculées par nyc_live._city() ; la somme zone
   par zone ne servait qu'à la case liée au panneau, et elle parcourait
   231 x 168 entiers par pas. */

/* La course du compteur et le décalage entre cases suivent le pas du cycle,
   au lieu d'être deux constantes.

   Le pas vaut 1 700 ms sur la journée et la semaine, 3 400 ms sur l'année.
   La semaine a compté 168 pas de 243 ms, et une course fixe de 420 ms n'y
   finissait jamais avant le pas suivant : le nombre restait une texture
   mouvante au lieu de se poser. Le plafond reste 420 ms, qui est le bon
   rythme quand le pas en laisse la place ; sous un pas de 700 ms, la course
   prend six dixièmes du pas et le nombre se lit encore. */
function rollMs() { return Math.min(420, META.cycles[S.scope].ms * 0.6); }
function stagger() { return Math.min(70, META.cycles[S.scope].ms * 0.12); }
const SOFT = window.matchMedia
  && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Dernière valeur atteinte par case, et animation en cours : sans la
   première, le compteur repartirait de l'arrivée précédente affichée, qui
   peut être un état intermédiaire si le pas change pendant la course. */
const LAST = {}, ANIM = {};

/* Compte de l'ancienne valeur à la nouvelle. `render` fait le texte, ce qui
   garde ici la même mise en forme qu'ailleurs — francs, miles, pourcentage —
   sans que cette fonction ait à savoir laquelle.

   `to` est un nombre, ou un tableau quand la case porte deux séries. Les
   deux termes comptent alors ensemble et la ligne arrive d'un seul
   mouvement ; recomposée terme par terme, elle se lirait comme deux cases
   mal alignées. */
/* `paint` remplace la pose du texte quand la case n'est pas un bout de
   typographie : un <text> de SVG, ou les segments du taximètre. Sans lui,
   rollCell ajoutait un <span> là où il n'a pas de sens. */
function rollCell(el, key, to, render, delay, paint) {
  let put;
  if (paint) {
    put = v => paint(el, render(v));
  } else {
    let span = el.firstElementChild;
    if (!span) { span = document.createElement("span"); el.appendChild(span); }
    put = v => { span.textContent = render(v); };
  }
  const many = Array.isArray(to);
  const from = LAST[key] === undefined ? to : LAST[key];
  LAST[key] = to;
  if (ANIM[key]) { cancelAnimationFrame(ANIM[key]); ANIM[key] = 0; }
  const same = many ? from.every((x, i) => x === to[i]) : from === to;
  if (same || SOFT) { put(to); return; }
  const mix = e => many ? from.map((x, i) => x + (to[i] - x) * e)
                        : from + (to - from) * e;
  put(from);
  const t0 = performance.now() + delay;
  /* La durée est figée au départ de la course : la relire à chaque trame
     ferait changer la vitesse en plein vol si le cycle change. */
  const ms = rollMs();
  const tick = now => {
    /* Avant t0 la fraction est nulle : c'est ainsi que se fait l'attente de
       la case, sans minuteur à annuler si le pas change entre-temps. */
    const u = Math.max(0, Math.min(1, (now - t0) / ms));
    const e = 1 - Math.pow(1 - u, 3);
    put(mix(e));
    ANIM[key] = u < 1 ? requestAnimationFrame(tick) : 0;
  };
  ANIM[key] = requestAnimationFrame(tick);
}

/* Remplace le texte d'une case en croisant l'ancien et le nouveau. Rien ne
   bouge si le texte est le même : la distance moyenne et la zone la plus
   active tiennent souvent plusieurs pas, et les faire clignoter sans qu'elles
   changent dirait un changement qui n'a pas eu lieu. */
/* `html` n'est vrai que pour la case des paliers, dont le grand nombre porte
   deux morceaux de typographie différente — le montant et sa plage horaire.
   Le texte, lui, reste posé en textContent partout ailleurs. */
function setCell(el, text, html) {
  const cur = el.lastElementChild;
  const same = cur && (html ? cur.innerHTML === text : cur.textContent === text);
  if (same) { return; }
  const next = document.createElement("span");
  if (html) { next.innerHTML = text; } else { next.textContent = text; }
  next.className = "in";
  el.appendChild(next);
  /* Deux images : la première pose l'état d'entrée, la seconde le relâche.
     Poser et retirer la classe dans la même image ne déclencherait aucune
     transition — le navigateur n'aurait jamais rendu l'état de départ. */
  requestAnimationFrame(() => requestAnimationFrame(() => {
    next.classList.remove("in");
    if (cur) {
      cur.classList.add("out");
      cur.addEventListener("transitionend", () => cur.remove(), { once: true });
      /* Filet : une transition sur un onglet caché ne se termine jamais, et
         les sortants s'empileraient derrière l'entrant. */
      setTimeout(() => cur.remove(), 600);
    }
  }));
}


/* Ce qui remplit chaque case. Les quatre objets ne partagent que leur titre ;
   leur corps n'a rien en commun, donc le dispatch est une table et non une
   suite de conditions. */
const PAINT = {
  mph: setDial,
  fare: setMeter,
  pax: (cell, step, delay) => setParts("pax", step, delay),
  dist: setNav,
};

function onState() {
  /* Les arcs de la jante suivent la même durée que les compteurs. */
  document.documentElement.style.setProperty("--arc", rollMs() + "ms");
  CELLS.forEach((key, rank) => {
    const cell = document.getElementById("c-" + key);
    const delay = rank * stagger();
    setCell(cell.querySelector(".k-lab"), cellLabel(key));
    PAINT[key](cell, S.step, delay);
  });
}
</script>
"""

# ---------------------------------------------------------------------------
# Composant 2 — la carte à pics
# ---------------------------------------------------------------------------
_MAP = r"""
<script src="https://unpkg.com/deck.gl@8.9.36/dist.min.js"></script>
<div id="map" style="position:absolute; inset:0;"></div>
<script>
let deckgl = null;

/* La carte a deux villes, une par thème, et le sol de chacune est celui de
   la page : le bloc n'est pas une fenêtre posée sur le tableau de bord, c'est
   le tableau de bord qui se soulève. Le socle est donc laissé au `C.bg` du
   socle commun, et ce qui change ici est la ville dessus.

   Les deux rampes finissent sur l'ambre, qui dit partout la valeur haute. Ce
   qui s'inverse est le bas : la nuit part du violet sourd et monte vers la
   lumière, le jour part de la pierre et descend vers l'ambre brûlé. Dans les
   deux cas la zone la plus active est la plus contrastée du sol — et le bas
   de la rampe reste à l'écart de ce sol, sans quoi les zones éteintes de
   trois heures du matin s'effaceraient au lieu de se montrer basses. */
const NIGHT = CFG.mode === "dark";

/* Hauteur de la plus haute colonne, en mètres. */
const TOP_M = 9000;

/* Les colonnes comptent des blocs, elles ne mesurent plus une proportion.
   Avant, la hauteur valait (valeur - minimum) / amplitude : une échelle
   relative, et donc illisible, parce que l'écart entre zones est énorme. Au
   pas le plus chargé du cycle horaire, le pic fait 465,3 courses/h et la
   zone médiane 0,75 — un rapport de 617. Sur neuf kilomètres de colonne,
   la zone médiane faisait onze mètres, et les deux cents zones du bas se
   confondaient en une croûte au ras du sol.

   Un bloc vaut donc une quantité nommée, et l'arrondi est fait au supérieur :
   toute zone qui a au moins une course garde un bloc entier. Rien ne
   disparaît sous le seuil du pixel, et l'œil compte au lieu d'estimer.

   Les trois unités sont choisies pour que le pic tienne autour de trente
   blocs dans les trois mesures — 33 pour les courses, 31 pour la recette,
   33 pour la distance. Un bloc garde ainsi la même hauteur quelle que soit
   la mesure et quel que soit le cycle, ce qu'une échelle relative interdit.

   La contrepartie est assumée : au pas le plus chargé, 208 zones sur 259 se
   retrouvent à un seul bloc. On gagne la visibilité du bas de l'échelle, on
   y perd la discrimination. C'est l'inverse exact du défaut précédent. */
const UNIT = { trips: 20, revenue: 1500 };
const UNIT_LABEL = { trips: "20 courses/h", revenue: "1 500 $/h" };
const MAX_BLOCKS = 34;
const BLOCK_M = TOP_M / MAX_BLOCKS;

function blocksOf(real, measure) {
  if (!(real > 0)) { return 0; }
  return Math.ceil(real / UNIT[measure]);
}

const RAMP = NIGHT ? { a: [58, 54, 67], b: [226, 172, 70] }
                   : { a: [168, 164, 178], b: [169, 116, 26] };

/* L'arête sépare deux faces de même teinte. De nuit elle est une lumière, de
   jour un trait d'ombre : la même arête en blanc disparaîtrait sur la pierre
   pâle. */
const EDGE = NIGHT ? [255, 255, 255, 30] : [28, 26, 32, 46];

/* Une lumière rasante donne le relief : c'est l'éclairage qui fait lire une
   hauteur, une couleur seule se lit comme une tache. L'effet est construit
   une fois deck.gl chargé, et seulement si la version en place l'expose. */
function lighting() {
  if (!deck.LightingEffect || !deck.DirectionalLight || !deck.AmbientLight) {
    return [];
  }
  /* De nuit, une rasante marquée : c'est elle qui sculpte la ville dans le
     noir. De jour, le ciel ouvert — l'ambiante porte, la rasante ne fait plus
     que poser les flancs, sinon les faces éclairées brûlent sur le sol pâle. */
  return [new deck.LightingEffect({
    amb: new deck.AmbientLight({ color: NIGHT ? [255, 248, 232] : [255, 253, 248],
                                 intensity: NIGHT ? 1.05 : 1.02 }),
    key: new deck.DirectionalLight({ color: NIGHT ? [255, 240, 214] : [255, 250, 238],
                                     intensity: NIGHT ? 1.9 : 0.95,
                                     direction: [-1.2, -2.6, -1.8] }),
  })];
}

/* Staten Island et Newark sont hors du cadre utile : ils étirent l'emprise
   d'un bon tiers pour une poignée de courses, et la ville se retrouverait
   deux fois trop petite au milieu du noir. */
const OFF_FRAME = new Set(["Staten Island", "EWR"]);

/* Le cadrage est calculé, pas écrit : le bloc change de largeur avec la
   grille et avec la fenêtre, et une échelle en dur se décroche aussitôt.
   fitBounds raisonne en vue de dessus et sans rotation, alors que la carte
   est inclinée à 50 degrés : le sol fuit vers l'horizon et le cadre couvre
   beaucoup plus de terrain qu'une vue de dessus, si bien que la ville
   flottait au milieu du noir. Les quatre cinquièmes de niveau repris sont la
   correction de cette perspective — pas une marge de confort. */
function fitted() {
  const el = document.getElementById("map");
  let w = 180, s = 90, e = -180, n = -90;
  for (const z of META.zones) {
    if (OFF_FRAME.has(z.borough)) { continue; }
    for (const pt of z.polygon) {
      if (pt[0] < w) { w = pt[0]; }
      if (pt[0] > e) { e = pt[0]; }
      if (pt[1] < s) { s = pt[1]; }
      if (pt[1] > n) { n = pt[1]; }
    }
  }
  const vp = new deck.WebMercatorViewport({
    width: el.clientWidth || 600, height: el.clientHeight || 400 });
  const fit = vp.fitBounds([[w, s], [e, n]], { padding: 0 });
  /* Le centre de l'emprise tombe à l'est de Manhattan — le Queens est large
     et tire la boîte de ce côté. Un demi-dixième de degré vers l'ouest et un
     peu vers le sud ramènent dans le cadre ce qui porte la lecture. */
  return { longitude: fit.longitude - 0.03, latitude: fit.latitude - 0.012,
           zoom: fit.zoom + 0.8, pitch: CFG.view.pitch,
           bearing: CFG.view.bearing };
}

function boot() {
  deckgl = new deck.DeckGL({
    container: "map",
    initialViewState: fitted(),
    controller: true,
    effects: lighting(),
    onHover: info => {
      if (!info || !info.object) { tipAt(0, 0, null); return; }
      const d = info.object;
      /* L'infobulle énonce l'unité : sans elle, une pile de blocs se compte
         mais ne se traduit pas. */
      tipAt(info.x, info.y, "<b>" + d.name + "</b><br>" + d.borough + " · "
            + fmt(scaled(d.i, S.step, S.scope, S.measure), S.measure)
            + "<br>" + d.n + (d.n === 1 ? " bloc" : " blocs")
            + " · 1 bloc = " + UNIT_LABEL[S.measure]);
    },
    layers: [],
  });
  /* Un ResizeObserver et non l'évènement `resize` de la fenêtre : au moment
     de boot(), l'iframe peut n'avoir aucune dimension, et le cadrage se
     ferait alors sur les valeurs de secours. L'observateur se déclenche dès
     qu'une taille réelle existe, puis à chaque changement. */
  const el = document.getElementById("map");
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      if (el.clientWidth && el.clientHeight) {
        deckgl.setProps({ initialViewState: fitted() });
      }
    }).observe(el);
  }
  frame();
}

function onState() { /* la boucle d'animation suit d'elle-même */ }

/* Les hauteurs sont interpolées entre le pas courant et le suivant : à un pas
   d'une seconde, une montée instantanée saccade. L'interpolation est faite
   ici et non par les transitions de deck.gl, qui repartiraient de zéro à
   chaque changement de cycle ou de mesure. */
function frame() {
  if (deckgl && META) {
    const t = phase();
    /* Le compte de blocs se prend sur la valeur interpolée, et non l'inverse :
       un bloc paraît ou s'efface quand la zone franchit son seuil, ce qui
       garde la pile entière à tout instant. Interpoler des hauteurs donnerait
       des fractions de bloc, donc des piles qu'on ne peut plus compter. */
    const sc = META.cycles[S.scope].scales[S.measure];
    const rows = META.zones.map((zone, i) => {
      const a = value(i, S.step, S.scope, S.measure);
      const b = value(i, S.step + 1, S.scope, S.measure);
      const n = blocksOf((a + (b - a) * t) * sc, S.measure);
      return { ...zone, i: i, n: n, v: Math.min(1, n / MAX_BLOCKS) };
    });
    deckgl.setProps({ layers: [new deck.PolygonLayer({
      id: "zones", data: rows, extruded: true, filled: true, pickable: true,
      getPolygon: d => d.polygon,
      getElevation: d => d.n * BLOCK_M,
      /* Rampe en puissance : la racine remonte les zones moyennes, qui
         resteraient autrement indistinctes du fond. Elle suit le compte de
         blocs, pour que la teinte et la hauteur disent la même chose. */
      getFillColor: d => {
        const u = Math.pow(d.v, 0.45);
        return [RAMP.a[0] + u * (RAMP.b[0] - RAMP.a[0]),
                RAMP.a[1] + u * (RAMP.b[1] - RAMP.a[1]),
                RAMP.a[2] + u * (RAMP.b[2] - RAMP.a[2]), 234];
      },
      getLineColor: EDGE,
      lineWidthMinPixels: 0.5,
      material: NIGHT ? { ambient: 0.38, diffuse: 0.78, shininess: 28,
                          specularColor: [72, 66, 54] }
                      : { ambient: 0.62, diffuse: 0.52, shininess: 14,
                          specularColor: [206, 202, 194] },
      updateTriggers: { getElevation: [S.scope, S.measure, S.step, t],
                        getFillColor: [S.scope, S.measure, S.step, t] },
    })] });
  }
  requestAnimationFrame(frame);
}
</script>
"""


# ---------------------------------------------------------------------------
# Composant 2 bis — la ville en maquette, un taxi par course
# ---------------------------------------------------------------------------
_CITY = r"""
<script src="https://unpkg.com/deck.gl@8.9.36/dist.min.js"></script>
<div id="city" style="position:absolute; inset:0;"></div>
<script>
/* Deux cent soixante et un gratte-ciel posent le sol, et au-dessus un taxi
   vaut une course. Le taxi part d'un endroit tiré au hasard dans sa zone de
   départ, rejoint un axe, traverse, puis se disperse vers un endroit tiré au
   hasard dans sa zone d'arrivée.

   L'axe n'est pas un décor : sans lui, mille directions partaient en même
   temps depuis mille points différents et la ville devenait une ruche. Les
   courses qui restent dans leur zone ne voyagent pas — un trait aurait
   inventé un trajet qu'on ne connaît pas. Elles se posent et font la
   densité.

   Les tours prennent la couleur du sol de leur secteur : le relief ne vient
   que de la lumière. Toute couleur chaude sur la carte entrerait en
   concurrence avec le jaune des taxis, qui est la seule donnée en mouvement.
*/
const NIGHT = CFG.mode === "dark";

/* À l'échelle, une tour de 400 m sur une vue qui porte 25 km fait un pixel.
   On garde les emprises et les hauteurs réelles mais on les grossit, comme
   une maquette. */
const EXAG = 3.2, FOOT = 2.4;
const ALT = 950;              /* altitude de vol, au ras des toits          */
/* Les arcs de flux partent sous les taxis : ils sont le dessous de la scène,
   la voiture est au-dessus. */
const ARC_Z = 420;
const REST = 70;              /* altitude des taxis posés, entre les tours  */
const SPEED = 0.055;          /* degrés par seconde : vitesse constante     */
const TAXI_PX = 6.2;

const PER_AXIS = 260;         /* courses par axe avant d'en ouvrir un autre */
const MAX_AXIS = 5;
const AXIS_GAP = 0.0062;
const JITTER = 0.0011;
const IN = 0.20, OUT = 0.80;  /* part du trajet passée à rejoindre l'axe    */
const KLAT = Math.cos(40.75 * Math.PI / 180);

/* Les sept secteurs dessinés. Staten Island en est absent — six courses par
   jour — et « Airports » aussi, mais pour une autre raison : JFK, LaGuardia
   et Newark sont à trente-cinq kilomètres les uns des autres, et un centre
   de gravité entre eux tomberait dans la baie. Le volume de ce secteur est
   donc réparti entre deux nœuds de dessin, JFK et LaGuardia, au prorata
   mesuré de chaque pas — il bouge : JFK prend 97 % des départs à 5 h et 40 %
   à 10 h. Newark n'en est pas : les taxis jaunes n'y chargent pas, trois
   courses sur toute la période. */
const SECTORS = [0, 1, 2, 3, 4, 5, 6];
let AIRSEC = -1, AIRNODE = [];

/* La position d'un nœud. Les neuf secteurs la lisent dans CITY.lonlat, les
   deux aéroports dans CITY.air : leurs indices prennent la suite. */
function nodeAt(i) {
  return i < CITY.lonlat.length ? CITY.lonlat[i]
                                : AIRNODE[i - CITY.lonlat.length].lonlat;
}

/* Un secteur développé en nœuds de dessin et en poids. Tout secteur vaut
   pour lui-même ; celui des aéroports vaut pour deux. */
function spread(s) {
  if (s !== AIRSEC || !AIRNODE.length) { return [[s, 1]]; }
  const sh = CITY.air.share[S.scope] || CITY.air.share.hour;
  const w = sh[((S.step % sh.length) + sh.length) % sh.length];
  const out = [];
  for (let k = 0; k < AIRNODE.length; k++) {
    if (w[k] > 0) { out.push([AIRNODE[k].pool, w[k]]); }
  }
  return out.length ? out : [[AIRNODE[0].pool, 1]];
}

/* Le ruban et la carte nomment les mêmes neuf zones : ils en prennent donc
   la même couleur, à deux intensités. Le ruban la porte pleine — entre neuf
   nœuds, la couleur est sa seule distinction. La carte la ramène vers le
   gris violacé de la maquette, parce que le jaune des taxis doit rester la
   seule couleur vive au-dessus du sol. Même teinte des deux côtés : on
   reconnaît une zone d'un bloc à l'autre sans relire son nom. */
const SLAB = [116, 112, 138];
const MUTE = 0.72;
let TINT = null;

function rgb(hex) {
  const h = String(hex).replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16),
          parseInt(h.slice(4, 6), 16)];
}
/* Le sol de la carte est le fond de la page : le bloc n'est pas une fenêtre
   posée sur le tableau de bord, c'est le tableau de bord qui se soulève. */
const SCENE_BG = rgb(C.bg);
const GROUND_A = 46 / 255;
const LIFT = NIGHT ? [152, 148, 174] : [198, 195, 210];
const LIFT_K = NIGHT ? [0.52, 0.34] : [0.26, 0.30];

/* Le taxi est dessiné dans un canvas, pas chargé : une image venue d'ailleurs
   aurait demandé un hôte, et un rond ne ressemble pas à une voiture. */
function taxiAtlas() {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#2a241c";
  g.fillRect(14, 15, 11, 6); g.fillRect(14, 43, 11, 6);
  g.fillRect(39, 15, 11, 6); g.fillRect(39, 43, 11, 6);
  g.fillStyle = "#edad12";
  g.beginPath();
  if (g.roundRect) { g.roundRect(8, 19, 48, 26, 8); }
  else { g.rect(8, 19, 48, 26); }
  g.fill();
  g.fillStyle = "#30281e";
  g.fillRect(41, 23, 10, 18);
  g.fillStyle = "#8a6f3a";
  g.fillRect(20, 25, 16, 14);
  g.fillStyle = "#fff3cf";
  g.fillRect(27, 29, 7, 6);
  return c;
}
const ATLAS = taxiAtlas();
const MAPPING = { taxi: { x: 0, y: 0, width: 64, height: 64,
                          anchorX: 32, anchorY: 32, mask: false } };

/* Les aéroports ne sont pas un secteur de la maquette : JFK et LaGuardia
   tombent dans le polygone du Queens, Newark est hors cadre. Mais le ruban
   les montre maintenant à part, et 7 % des courses y passent — un repère
   sur la carte est ce qui relie les deux lectures. Un anneau, trois lettres,
   dans la couleur du nœud. */
const AIR_CODE = { "JFK Airport": "JFK", "LaGuardia Airport": "LGA",
                   "Newark Airport": "EWR" };
let AIRPORTS = [], AIR_RGB = [120, 120, 130];

let CITY = null, POOL = null, GROUND = null, SPACED = null;
let MOV = null, SIT = null, AXES = [], ARCS = [], STATIC = [];
let deckgl = null, VIEWSTATE = null, LABEL_POS = null, started = 0;

function pick(s) { return POOL[s][(Math.random() * POOL[s].length) | 0]; }

/* Un tableau de flux par cycle, et non les seules vingt-quatre heures.
   Jusqu'ici le jour et le mois n'avaient que leur moyenne — un seul nombre,
   le même aux sept ou douze pas : le nombre de taxis ne bougeait pas d'un
   lundi à un samedi.

   L'unité est la course par heure sur les trois cycles. Un pas de mois
   compte un mois entier, vingt-quatre fois un pas d'heure ; à un taxi par
   course, l'année aurait affiché 120 000 voitures contre 5 000 à l'heure.
   scripts/J_build_cycles.py divise donc les pas de mois par vingt-quatre à
   l'écriture des agrégats, et un taxi vaut partout la même chose : une
   course de l'heure moyenne du pas. */
function matrix() {
  const m = CITY.trips[S.scope] || CITY.trips.hour;
  return m[((S.step % m.length) + m.length) % m.length];
}

/* Les liaisons retenues, dans l'ordre décroissant, avec leur épaisseur et
   leur couleur. L'épaisseur suit la racine du volume : une liaison dix fois
   plus forte est trois fois plus épaisse, ce qui est la règle d'un tracé de
   flux — l'oeil compare des surfaces, pas des largeurs. */
function arcTop(all) {
  if (!all.length) { return []; }
  let max = 0;
  for (const q of all) { if (q.n > max) { max = q.n; } }
  const keep = all.filter(q => q.n >= max * ARC_MIN)
    .sort((x, y) => y.n - x.n).slice(0, ARC_KEEP);
  return keep.map(q => {
    const A = nodeAt(q.a), B = nodeAt(q.b);
    const c = TINT[secOf(q.a)] || [120, 120, 130];
    return { s: [A[0], A[1], ARC_Z], t: [B[0], B[1], ARC_Z],
             w: 0.7 + 3.1 * Math.sqrt(q.n / max),
             c: [c[0], c[1], c[2]] };
  });
}

function axesFor(a, b, n) {
  const A = nodeAt(a), B = nodeAt(b);
  const dx = (B[0] - A[0]) * KLAT, dy = B[1] - A[1];
  const len = Math.hypot(dx, dy) || 1;
  const px = -dy / len / KLAT, py = dx / len;
  const out = [];
  for (let k = 0; k < n; k++) {
    const off = (k - (n - 1) / 2) * AXIS_GAP;
    out.push([[A[0] + (B[0] - A[0]) * 0.17 + px * off,
               A[1] + (B[1] - A[1]) * 0.17 + py * off],
              [A[0] + (B[0] - A[0]) * 0.83 + px * off,
               A[1] + (B[1] - A[1]) * 0.83 + py * off],
              [px, py]]);
  }
  return out;
}

/* À quel secteur appartient un nœud. Les nœuds d'aéroport prennent la suite
   des secteurs dessinés, et ils appartiennent tous au secteur des
   aéroports. */
function secOf(i) {
  return i < CITY.lonlat.length ? i : AIRSEC;
}

/* Les liaisons gardées pour le tracé : les plus fortes, jusqu'à quarante-
   huit, et au moins un cinquantième de la plus forte.

   Tout tracer faisait soixante-douze arcs sur neuf nœuds, donc un écheveau
   où l'on ne lisait plus une seule liaison. Le seuil et le plafond laissent
   voir la structure — Midtown au centre de tout, les deux aéroports vers
   Manhattan — sans la noyer. */
const ARC_KEEP = 48, ARC_MIN = 0.02;

function build() {
  const m = matrix();
  const o = [], e1 = [], e2 = [], d = [], inv = [], ph = [];
  const rp = [];
  const arcs = [];
  AXES = [];
  /* Les secteurs de départ et d'arrivée possibles : les sept dessinés, plus
     celui des aéroports, que spread() développera en deux nœuds. */
  const SRC = AIRSEC >= 0 ? SECTORS.concat([AIRSEC]) : SECTORS;
  for (const sa of SRC) {
    for (const sb of SRC) {
      if (!(m[sa][sb] > 0)) { continue; }
      for (const [a, wa] of spread(sa)) {
        for (const [b, wb] of spread(sb)) {
          const n = Math.round(m[sa][sb] * wa * wb);
          if (n <= 0) { continue; }
          /* a === b et non sa === sb : une course JFK → LaGuardia part bien
             d'un nœud pour un autre, alors qu'elle reste dans le secteur. */
          if (a === b) {
            for (let k = 0; k < n; k++) {
              const p = pick(a);
              rp.push(p[0], p[1], REST);
            }
            continue;
          }
          arcs.push({ a: a, b: b, n: n });
          const na = Math.max(1, Math.min(MAX_AXIS, Math.round(n / PER_AXIS)));
          const ax = axesFor(a, b, na);
          for (const q of ax) {
            AXES.push({ s: [q[0][0], q[0][1], ALT],
                        t: [q[1][0], q[1][1], ALT] });
          }
          for (let k = 0; k < n; k++) {
            const q = ax[(Math.random() * na) | 0];
            const j = (Math.random() - 0.5) * 2 * JITTER;
            const P = pick(a), Q = pick(b);
            const x1 = q[0][0] + q[2][0] * j, y1 = q[0][1] + q[2][1] * j;
            const x2 = q[1][0] + q[2][0] * j, y2 = q[1][1] + q[2][1] * j;
            const L = Math.hypot((x1 - P[0]) * KLAT, y1 - P[1])
                    + Math.hypot((x2 - x1) * KLAT, y2 - y1)
                    + Math.hypot((Q[0] - x2) * KLAT, Q[1] - y2) || 0.01;
            o.push(P[0], P[1]); e1.push(x1, y1); e2.push(x2, y2);
            d.push(Q[0], Q[1]);
            inv.push(SPEED / L); ph.push(Math.random());
          }
        }
      }
    }
  }
  LABEL_POS = labelSpots();
  ARCS = arcTop(arcs);
  const nm = inv.length, nr = rp.length / 3;
  MOV = { n: nm, pos: new Float32Array(nm * 3), ang: new Float32Array(nm),
          o: new Float32Array(o), e1: new Float32Array(e1),
          e2: new Float32Array(e2), d: new Float32Array(d),
          inv: new Float32Array(inv), ph: new Float32Array(ph) };
  const ra = new Float32Array(nr);
  for (let i = 0; i < nr; i++) { ra[i] = Math.random() * 360; }
  SIT = { n: nr, pos: new Float32Array(rp), ang: ra };
  STATIC = staticLayers();
  sayCount(nm + nr);
}

/* Le compte des taxis à l'écran, et ce qu'un taxi vaut.

   Sans le nombre, on ne sait pas si la ville s'est vidée ou si le cadrage a
   changé, et on ne peut pas comparer deux pas de mémoire.

   La clé est posée sur la carte, dans la bande des couleurs de secteurs, et
   non sous le titre. Une clé de carte se lit en regardant la carte : à côté
   du titre, il fallait quitter l'image pour savoir ce qu'on regardait, et le
   titre grandissait d'une ligne.

   Un taxi vaut une course, et c'est le compte qui est dit en premier : ce
   qu'on veut savoir est combien de courses la ville fait à cette heure, et
   l'équivalence vient après, pour expliquer ce qu'on voit bouger.

   La ligne a dit « un taxi = une course par heure », ce qui ne se comprenait
   pas : « par heure » qualifie le compte, pas le taxi. Elle a ensuite compté
   les voitures en circulation — courses de l'heure fois durée du trajet — ce
   qui était juste mais répondait à une question que personne ne posait, et
   enlevait du monde à l'écran. Le compte des courses est la donnée ; la
   voiture n'est que le marqueur qui la porte. */
function cabKey() {
  const el = document.getElementById("legend");
  if (!el || !CFG.legend) { return null; }
  let k = el.querySelector("#cabkey");
  if (!k) {
    k = document.createElement("span");
    k.id = "cabkey";
    k.style.fontWeight = "600";
    el.insertBefore(k, el.firstChild);
  }
  return k;
}
function sayCount(n) {
  const k = cabKey();
  if (!k) { return; }
  k.textContent = FR(n) + " courses dans l'heure · un taxi = une course";
}

/* Trois segments : décollage vers l'axe, traversée à altitude constante,
   dispersion vers le point d'arrivée. L'altitude suit, sinon les taxis
   coupaient à travers les tours au départ et à l'arrivée. */
function advance(t) {
  const P = MOV.pos, O = MOV.o, A = MOV.e1, B = MOV.e2, D = MOV.d;
  const I = MOV.inv, H = MOV.ph, G = MOV.ang, DEG = 180 / Math.PI;
  for (let i = 0; i < MOV.n; i++) {
    let u = H[i] + t * I[i];
    u -= Math.floor(u);
    const j = 2 * i, k = 3 * i;
    if (u < IN) {
      const v = u / IN;
      P[k] = O[j] + (A[j] - O[j]) * v;
      P[k + 1] = O[j + 1] + (A[j + 1] - O[j + 1]) * v;
      P[k + 2] = REST + (ALT - REST) * v;
      G[i] = Math.atan2(A[j + 1] - O[j + 1], (A[j] - O[j]) * KLAT) * DEG;
    } else if (u < OUT) {
      const v = (u - IN) / (OUT - IN);
      P[k] = A[j] + (B[j] - A[j]) * v;
      P[k + 1] = A[j + 1] + (B[j + 1] - A[j + 1]) * v;
      P[k + 2] = ALT;
      G[i] = Math.atan2(B[j + 1] - A[j + 1], (B[j] - A[j]) * KLAT) * DEG;
    } else {
      const v = (u - OUT) / (1 - OUT);
      P[k] = B[j] + (D[j] - B[j]) * v;
      P[k + 1] = B[j + 1] + (D[j + 1] - B[j + 1]) * v;
      P[k + 2] = ALT + (REST - ALT) * v;
      G[i] = Math.atan2(D[j + 1] - B[j + 1], (D[j] - B[j]) * KLAT) * DEG;
    }
  }
}

/* Les étiquettes sont ramenées dans le cadre en pixels, pas en degrés : la
   vue est tournée de 29 degrés, si bien qu'une borne en longitude ne
   correspond à aucun bord de l'écran et le nom de Brooklyn sortait quand
   même. On projette, on borne, on reprojette. */
function labelSpots() {
  const el = document.getElementById("city");
  const W = el.clientWidth || 600, H = el.clientHeight || 400;
  if (!VIEWSTATE || !deck.WebMercatorViewport) {
    return SECTORS.map(a => CITY.lonlat[a]);
  }
  const vp = new deck.WebMercatorViewport(
    Object.assign({ width: W, height: H }, VIEWSTATE));
  const padX = Math.min(64, W * 0.14), padY = Math.min(28, H * 0.07);
  /* Le nom est posé en altitude : sa place à l'écran n'est pas celle du sol
     sous lui. On borne la position affichée, puis on retire ce décalage pour
     retrouver le point de sol à donner à la couche. */
  const z = ALT + 340;
  return SECTORS.map(a => {
    const g = CITY.lonlat[a];
    const p0 = vp.project([g[0], g[1], 0]);
    const p1 = vp.project([g[0], g[1], z]);
    const x = Math.min(W - padX, Math.max(padX, p1[0]));
    const y = Math.min(H - padY, Math.max(padY, p1[1]));
    return vp.unproject([x - (p1[0] - p0[0]), y - (p1[1] - p0[1])]);
  });
}

function staticLayers() {
  const out = [];
  out.push(new deck.PolygonLayer({
    id: "ground", data: CITY.polys, extruded: false, filled: true,
    stroked: false, pickable: false,
    getPolygon: d => d.p,
    /* Le sol porte la couleur du nœud correspondant dans le ruban, assez
       franchement pour qu'on fasse le rapprochement : un bloc dit quelles
       liaisons, l'autre où elles tombent, et c'est la couleur qui les relie.
       Elle reste sous le jaune des taxis, qui est la donnée en mouvement. */
    getFillColor: d => [TINT[d.s][0], TINT[d.s][1], TINT[d.s][2], 58],
  }));
  out.push(new deck.PolygonLayer({
    id: "buildings", data: CITY.bld, extruded: true, filled: true,
    pickable: false,
    getPolygon: d => d.p,
    getElevation: d => d.h * EXAG,
    getFillColor: d => {
      const g = GROUND[d.s], t = Math.min(1, (d.h - 61) / 240);
      const k = LIFT_K[0] + LIFT_K[1] * t;
      return [g[0] + (LIFT[0] - g[0]) * k, g[1] + (LIFT[1] - g[1]) * k,
              g[2] + (LIFT[2] - g[2]) * k, 248];
    },
    /* L'ambiante et la rasante additionnées dépassaient 1 : les toits
       saturaient en blanc quelle que soit leur couleur de base. */
    material: { ambient: 0.70, diffuse: 0.62, shininess: 10,
                specularColor: [30, 28, 38] },
  }));
  out.push(new deck.LineLayer({
    id: "axes", data: AXES,
    getSourcePosition: d => d.s, getTargetPosition: d => d.t,
    getColor: NIGHT ? [160, 156, 182, 20] : [118, 114, 140, 20], getWidth: 1,
  }));
  /* Les flux de secteur à secteur, en arcs. La maquette montrait les courses
     une par une, en voitures, et une voiture ne dit rien d'une liaison : il
     en faut cent pour voir un courant, et à cent voitures on ne voit plus
     lesquelles vont ensemble. L'arc dit la liaison, la voiture dit la course,
     et les deux sont la même donnée à deux échelles.

     L'arc part du secteur de départ et prend sa couleur : un flux a un sens,
     et c'est l'origine qui le donne. */
  out.push(new deck.ArcLayer({
    id: "flux", data: ARCS, pickable: false,
    getSourcePosition: d => d.s, getTargetPosition: d => d.t,
    getSourceColor: d => [d.c[0], d.c[1], d.c[2], NIGHT ? 150 : 130],
    getTargetColor: d => [d.c[0], d.c[1], d.c[2], NIGHT ? 36 : 30],
    getWidth: d => d.w, widthUnits: "pixels",
    /* À plat. Une hauteur d'arc faisait voûter les liaisons au-dessus de la
       ville, et ces voûtes se lisaient comme des trajectoires de courses,
       qui sont au sol. Une liaison n'est pas un trajet : c'est une quantité
       entre deux endroits, et elle se tient sur la carte. */
    getHeight: 0, greatCircle: false,
    parameters: { depthTest: false },
  }));
  /* Sans depthTest coupé, les tours mangeaient la première lettre : le texte
     passait derrière une façade plus haute que son altitude. */
  out.push(new deck.TextLayer({
    id: "zone-text", data: SECTORS,
    getPosition: a => {
      const p = LABEL_POS[a] || CITY.lonlat[a];
      return [p[0], p[1], ALT + 340];
    },
    getText: a => SPACED[a], getSize: 10.5, sizeUnits: "pixels",
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif", fontWeight: 700,
    fontSettings: { sdf: true, fontSize: 64, buffer: 10 },
    outlineWidth: 3.4,
    outlineColor: NIGHT ? [SCENE_BG[0], SCENE_BG[1], SCENE_BG[2], 215]
                        : [SCENE_BG[0], SCENE_BG[1], SCENE_BG[2], 222],
    getColor: NIGHT ? [214, 210, 230, 235] : [72, 68, 94, 230],
    parameters: { depthTest: false },
  }));
  if (AIRPORTS.length) {
    out.push(new deck.ScatterplotLayer({
      id: "air-ring", data: AIRPORTS, pickable: false,
      getPosition: d => [d.lon, d.lat, ALT + 160],
      radiusUnits: "pixels", getRadius: 4.6,
      stroked: true, filled: false, lineWidthUnits: "pixels",
      getLineWidth: 1.5,
      getLineColor: [AIR_RGB[0], AIR_RGB[1], AIR_RGB[2], 225],
      parameters: { depthTest: false },
    }));
    out.push(new deck.TextLayer({
      id: "air-text", data: AIRPORTS, pickable: false,
      getPosition: d => [d.lon, d.lat, ALT + 160],
      getText: d => d.code, getSize: 9, sizeUnits: "pixels",
      getPixelOffset: [0, -14],
      fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif",
      fontWeight: 700,
      fontSettings: { sdf: true, fontSize: 64, buffer: 10 },
      outlineWidth: 3.2,
      outlineColor: [SCENE_BG[0], SCENE_BG[1], SCENE_BG[2], 222],
      getColor: [AIR_RGB[0], AIR_RGB[1], AIR_RGB[2], 245],
      parameters: { depthTest: false },
    }));
  }
  return out;
}

function layers() {
  const out = STATIC.slice();
  /* billboard: false pose la voiture à plat sur le plan du sol : elle tourne
     avec la carte au lieu de rester face à l'écran, et son cap se lit. */
  out.push(new deck.IconLayer({
    id: "sit", iconAtlas: ATLAS, iconMapping: MAPPING,
    data: { length: SIT.n,
            attributes: { getPosition: { value: SIT.pos, size: 3 },
                          getAngle: { value: SIT.ang, size: 1 } } },
    getIcon: () => "taxi", sizeUnits: "pixels", getSize: TAXI_PX * 0.78,
    billboard: false, opacity: 0.62,
  }));
  out.push(new deck.IconLayer({
    id: "mov", iconAtlas: ATLAS, iconMapping: MAPPING,
    data: { length: MOV.n,
            attributes: { getPosition: { value: MOV.pos, size: 3 },
                          getAngle: { value: MOV.ang, size: 1 } } },
    getIcon: () => "taxi", sizeUnits: "pixels", getSize: TAXI_PX,
    billboard: false,
  }));
  return out;
}

/* Le cadrage est calculé et non demandé à fitBounds : celui-ci raisonne en
   vue de dessus, et sous 54 degrés d'inclinaison la ville sortait trois fois
   trop petite. On cadre sur les centres de secteurs, pas sur les polygones :
   la pointe du Queens tirait l'emprise si loin que Manhattan tombait à
   quelques pixels. */
function fitted() {
  const el = document.getElementById("city");
  const W = el.clientWidth || 600;
  let w = 180, s = 90, e = -180, n = -90;
  /* Les nœuds d'aéroport entrent dans l'emprise : JFK est à l'est du Queens,
     et sans lui dans le calcul les taxis qui s'y posent tombaient hors
     cadre. Le cadrage recule de quelques pour cent, c'est le prix. */
  const pts = SECTORS.map(a => CITY.lonlat[a])
                     .concat(AIRNODE.map(d => d.lonlat));
  for (const c of pts) {
    if (c[0] < w) { w = c[0]; }
    if (c[0] > e) { e = c[0]; }
    if (c[1] < s) { s = c[1]; }
    if (c[1] > n) { n = c[1]; }
  }
  const mx = (e - w) * 0.17, my = (n - s) * 0.17;
  w -= mx; e += mx; s -= my; n += my;
  VIEWSTATE = { longitude: (w + e) / 2 - 0.022, latitude: (s + n) / 2 - 0.008,
                zoom: Math.log2(0.98 * W * 360 / (512 * (e - w))),
                pitch: 54, bearing: -29 };
  return VIEWSTATE;
}

/* Les emprises sont dilatées autour de leur centre : la hauteur grossit, pas
   la largeur, et les tours sortaient comme des aiguilles. */
function fatten() {
  for (const d of CITY.bld) {
    let cx = 0, cy = 0;
    for (const p of d.p) { cx += p[0]; cy += p[1]; }
    cx /= d.p.length; cy /= d.p.length;
    d.p = d.p.map(p => [cx + (p[0] - cx) * FOOT, cy + (p[1] - cy) * FOOT]);
  }
}

const SHORT = ["Downtown", "Midtown", "Uptown", "Harlem", "Queens",
               "Brooklyn", "Bronx", "SI", "EWR"];

async function boot() {
  CITY = await (await fetch(staticBase() + "nyc_city.json")).json();
  POOL = CITY.pools.pt.map(flat => {
    const a = new Array(flat.length / 2);
    for (let i = 0; i < a.length; i++) {
      a[i] = [CITY.pools.b[0] + flat[2 * i] / CITY.pools.sc,
              CITY.pools.b[1] + flat[2 * i + 1] / CITY.pools.sc];
    }
    return a;
  });
  TINT = CITY.order.map(name => {
    const c = rgb(META.areaColors[name] || C.dim);
    return [SLAB[0] + (c[0] - SLAB[0]) * MUTE,
            SLAB[1] + (c[1] - SLAB[1]) * MUTE,
            SLAB[2] + (c[2] - SLAB[2]) * MUTE];
  });
  GROUND = TINT.map(c => [
    SCENE_BG[0] + (c[0] - SCENE_BG[0]) * GROUND_A,
    SCENE_BG[1] + (c[1] - SCENE_BG[1]) * GROUND_A,
    SCENE_BG[2] + (c[2] - SCENE_BG[2]) * GROUND_A]);
  SPACED = SHORT.map(n => n.toUpperCase().split("").join(" "));
  AIRPORTS = META.zones
    .filter(z => AIR_CODE[z.name])
    .map(z => ({ code: AIR_CODE[z.name], lon: z.lon, lat: z.lat }));
  AIR_RGB = rgb((META.areaColors || {})["Airports"] || C.dim);
  if (CITY.air) {
    AIRSEC = CITY.air.sector;
    AIRNODE = CITY.air.node;
  }
  fatten();
  deckgl = new deck.DeckGL({
    container: "city",
    initialViewState: fitted(),
    /* La molette appartient à la page : un bloc posé dans une grille qui
       défile ne doit pas happer le défilement. */
    controller: { scrollZoom: false, dragPan: true, dragRotate: true,
                  doubleClickZoom: true, touchZoom: true, keyboard: false },
    effects: (deck.LightingEffect && deck.DirectionalLight
              && deck.AmbientLight)
      ? [new deck.LightingEffect({
          amb: new deck.AmbientLight({ color: [246, 244, 252],
                                       intensity: 0.95 }),
          key: new deck.DirectionalLight({ color: [248, 246, 255],
                 intensity: 0.55, direction: [-1.1, -2.2, -2.0] }) })]
      : [],
    layers: [],
  });
  const el = document.getElementById("city");
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      if (el.clientWidth && el.clientHeight) {
        deckgl.setProps({ initialViewState: fitted() });
        build();
      }
    }).observe(el);
  }
  build();
  frame();
}

function onState() { if (CITY) { build(); } }

function frame(now) {
  if (deckgl && MOV) {
    if (!started) { started = now || 0; }
    advance(((now || 0) - started) / 1000);
    deckgl.setProps({ layers: layers() });
  }
  requestAnimationFrame(frame);
}
</script>
"""


# ---------------------------------------------------------------------------
# Composant 3 — le ruban de barres qui défile
# ---------------------------------------------------------------------------
_PROFILE = r"""
<canvas id="cv" style="position:absolute; inset:0; width:100%; height:100%;"></canvas>
<script>
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
/* Le ruban montre une fenêtre glissante autour du pas courant. Le passé file
   vers la gauche, l'à-venir arrive par la droite ; le présent reste sur une
   ligne fixe, légèrement à gauche du centre pour laisser voir ce qui vient. */
/* Le présent reste sur une ligne fixe, légèrement à gauche du centre pour
   laisser voir ce qui vient. */
const NOW_X = 0.42;

/* La fenêtre s'accorde à la longueur du cycle. Seize cases pour sept jours
   feraient défiler la semaine deux fois et demie dans le cadre, et les
   libellés se chevaucheraient ; quelques cases de plus que le cycle suffisent
   à montrer que le ruban boucle. */
function window_() {
  const n = stepsOf(S.scope);
  const span = Math.min(16, n + 4);
  const behind = Math.round(NOW_X * span);
  return { span: span, behind: behind, ahead: span - behind };
}

function size() {
  const r = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * r;
  cv.height = cv.clientHeight * r;
  ctx.setTransform(r, 0, 0, r, 0, 0);
}
window.addEventListener("resize", size);

/* Quel pas se trouve sous une abscisse. C'est l'inverse exact du placement
   de `draw` : x = (NOW_X * span + k - t) * pitch, donc k s'en déduit. Les
   deux formules doivent rester d'accord — si l'une bouge, l'autre suit. */
function stepAt(px) {
  const { span } = window_();
  const pitch = cv.clientWidth / span;
  const n = stepsOf(S.scope);
  const k = Math.round(px / pitch - NOW_X * span + phase());
  return ((S.step + k) % n + n) % n;
}

function boot() {
  size();
  /* Le ruban devient une règle : on y désigne l'heure qu'on veut voir au lieu
     de l'attendre. Le maître de l'horloge est seul à tenir le pas, on lui
     demande donc d'y aller plutôt que de le poser soi-même — deux composants
     qui écrivent le même pas le feraient diverger. */
  cv.style.cursor = "pointer";
  cv.addEventListener("click", e => {
    if (!META) { return; }
    channel.postMessage({
      type: "seek", step: stepAt(e.clientX - cv.getBoundingClientRect().left) });
  });
  onState();
  frame();
}

/* Le titre dit ce qui est tracé, et seulement cela. Il change avec la
   mesure allumée, puisque c'est elle qui change ce qui est tracé. */
function onState() { headTitle(MEASURE_TITLE[S.measure]); }

function totals(step) {
  /* Somme sur toutes les zones : le ruban parle de la ville entière, la
     carte parle de chaque quartier. Les deux mesures sont des quantités, et
     une somme de quantités a un sens.

     La repondération par les courses qui vivait ici n'a plus d'objet : elle
     ne servait qu'à la distance moyenne, qu'additionner donnait 573,7 mi à
     5 h — la somme de quatre-vingt-dix-huit moyennes de zones, un nombre
     sans unité, là où la ville roule 5,65 mi par course. */
  const n = stepsOf(S.scope);
  const idx = ((step % n) + n) % n;
  const a = SERIES[S.scope][S.measure];
  let s = 0;
  for (let z = 0; z < Z; z++) { s += a[z * n + idx]; }
  return s;
}

let cache = { key: "", max: 1, vals: [] };
function series() {
  const key = S.scope + "/" + S.measure;
  if (cache.key !== key) {
    const n = stepsOf(S.scope);
    const vals = [];
    for (let i = 0; i < n; i++) { vals.push(totals(i)); }
    cache = { key: key, vals: vals, max: Math.max(...vals) };
  }
  return cache;
}

function frame() {
  if (META) { draw(); }
  requestAnimationFrame(frame);
}

/* Un entier pseudo-aléatoire mais stable, dérivé des seuls indices. Les
   fenêtres allumées d'une tour ne doivent pas changer d'une trame à
   l'autre : avec Math.random() le ruban aurait clignoté soixante fois par
   seconde. Même barre, mêmes fenêtres, toujours. */
function hash(a, b, c) {
  let h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return (h ^ (h >>> 16)) >>> 0;
}

/* Deux peaux de barre, une par mesure. L'objet dit la mesure avant que le
   titre soit lu : une tour pour les courses, une liasse pour les revenus.

   Règle qui ne se négocie pas : le sommet de la silhouette reste à `bh`
   dans les deux cas. Les retraits d'une tour se prennent sur la largeur, et
   le billet du haut d'une liasse se cale sur le sommet et non sur le socle.
   Une barre qui mentirait d'un pixel sur sa hauteur ne serait plus une
   barre, et le ruban se lit en hauteur. */
const SETBACK_H = 46, SPIRE_H = 112;

/* Le pas d'étage, le même pour toutes les tours du ruban, et la lumière.

   La lumière vient de la gauche, comme pour la jante du volant et le relief
   des panneaux : une seule source dans tout le tableau de bord, sinon chaque
   objet est éclairé par son propre soleil. La face droite est donc dans
   l'ombre et le toit prend la lumière.

   Les deux voiles sont du noir et du blanc transparents et non des couleurs
   calculées : une barre peut être à l'encre ou en gris de fond, et un voile
   transparent marche sur les deux sans qu'il faille mélanger des teintes
   dans le canvas. */
const FLOOR = 5;
const NIGHT = CFG.mode === "dark";

/* La profondeur. Le ruban passe de la face plate au volume : chaque barre
   gagne une joue à droite et un toit au-dessus, décalés d'un vecteur
   constant. Constant est le mot — c'est une axonométrie et non une
   perspective, donc deux barres de même hauteur font la même hauteur à
   l'écran où qu'elles soient dans la file, et la hauteur reste la mesure.
   Un point de fuite, lui, aurait rétréci les barres du fond.

   Le vecteur monte vers la droite : la lumière vient de la gauche dans tout
   le tableau de bord, donc la joue visible est celle de l'ombre et le toit
   prend la lumière. Neuf pixels de côté sur une barre de vingt-six, c'est
   le tiers — assez pour que le volume se voie, assez peu pour que la file
   reste une file et non un empilement. */
const DEPTH_X = 9, DEPTH_Y = -5;
function quad(p) {
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  for (let i = 2; i < p.length; i += 2) { ctx.lineTo(p[i], p[i + 1]); }
  ctx.closePath();
  ctx.fill();
}
const CORNICE = NIGHT ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.4)";
/* La masse est presque opaque. À huit dixièmes, l'horizon se lisait au
   travers des tours et les deux plans se mélangeaient : une tour qui laisse
   voir ce qui est derrière elle n'est pas une tour. Un vingtième suffit à
   l'asseoir dans la scène. */
const MASS_A = 0.95;

/* Le gris des tours est bleuté : c'est la couleur d'un volume de béton vu à
   distance, où l'air ajoute du bleu, et c'est aussi ce qui le distingue du
   gris neutre des filets et du texte. Le pas courant est plus sombre et plus
   dense, le reste recule.

   Le vitrage est bleu clair en thème clair — du verre réfléchissant le ciel —
   et jaune franc en thème sombre, où ce sont des fenêtres allumées dans la
   nuit. C'est la même façade à deux heures du jour. */
const MASS_HERE = NIGHT ? "#e4ebf7" : "#39465c";
const MASS_FAR = NIGHT ? "#5f6a7d" : "#95a1b5";
const GLASS = NIGHT ? "#ffd84d" : "#8fb6e3";

/* La tour : un corps jusqu'à l'épaule, un étage plus étroit au-dessus, une
   flèche sur les plus hautes, et des fenêtres sur le corps. C'est la
   silhouette à retraits que le zonage de 1916 a imposée à Manhattan — la
   raison pour laquelle les tours de New York sont des escaliers. */
function tower(x0, base, bw, bh, idx, col, lit) {
  const shoulder = bh > SETBACK_H ? bh * 0.72 : bh;
  /* Un bloc de la tour : la masse, sa face d'ombre à droite, sa corniche en
     haut. Trois passes sur le même rectangle, et c'est tout le volume. */
  const a = ctx.globalAlpha;
  /* Un bloc de la tour : la masse, le modelé en travers, la corniche en
     haut.

     Le modelé est un dégradé et non une face d'ombre à bord franc. Le bord
     franc faisait une arête à soixante-douze pour cent de la largeur, donc
     un pli de papier plié, et tout le ruban prenait l'air d'un dessin à la
     case. Un dégradé clair à gauche, neutre au milieu, sombre à droite est
     ce que fait la lumière sur une masse, et il n'a pas de bord. */
  const block = (x, y, w, hh) => {
    /* La joue et le toit passent avant la face : la face est le plan le
       plus proche, elle recouvre ce qui dépasse. */
    const dx = DEPTH_X, dy = DEPTH_Y;
    const joue = [x + w, y, x + w + dx, y + dy,
                  x + w + dx, y + dy + hh, x + w, y + hh];
    const toit = [x, y, x + dx, y + dy, x + w + dx, y + dy, x + w, y];
    ctx.globalAlpha = a * MASS_A;
    ctx.fillStyle = col;
    quad(joue);
    ctx.globalAlpha = a;
    ctx.fillStyle = "rgba(0,0,0," + (NIGHT ? 0.34 : 0.2) + ")";
    quad(joue);
    ctx.globalAlpha = a * MASS_A;
    ctx.fillStyle = col;
    quad(toit);
    ctx.globalAlpha = a;
    ctx.fillStyle = "rgba(255,255,255," + (NIGHT ? 0.1 : 0.22) + ")";
    quad(toit);
    ctx.globalAlpha = a * MASS_A;
    ctx.fillStyle = col;
    ctx.fillRect(x, y, w, hh);
    ctx.globalAlpha = a;
    const g = ctx.createLinearGradient(x, 0, x + w, 0);
    g.addColorStop(0, "rgba(255,255,255," + (NIGHT ? 0.07 : 0.14) + ")");
    g.addColorStop(0.52, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,0," + (NIGHT ? 0.34 : 0.2) + ")");
    ctx.fillStyle = g;
    ctx.fillRect(x, y, w, hh);
    if (hh > 4) {
      ctx.fillStyle = CORNICE;
      ctx.fillRect(x, y, w, 1);
    }
  };
  block(x0, base - shoulder, bw, shoulder);
  if (bh > SETBACK_H) {
    const w2 = Math.max(3, bw * 0.62);
    const h2 = bh > SPIRE_H ? bh * 0.92 : bh;
    block(x0 + (bw - w2) / 2, base - h2, w2, h2 - shoulder);
    if (bh > SPIRE_H) {
      const w3 = Math.max(2, bw * 0.3);
      block(x0 + (bw - w3) / 2, base - bh, w3, bh - h2);
    }
  }
  /* Le vitrage ne tient pas sous onze pixels de large : il y devient un
     bruit qui salit la couleur de la barre. */
  if (bw < 11 || shoulder < 14) { return; }
  /* Le vitrage est en bandeaux d'étage, et non en petits carrés.

     Les carrés de deux pixels et demi sur une grille de cinq étaient de la
     mosaïque : à cette taille l'oeil ne voit pas des fenêtres, il voit des
     pixels, et une tour de bureaux n'est de toute façon pas un damier — elle
     est une pile de bandeaux vitrés que les trumeaux coupent à peine.

     Trois passes par étage : le bandeau de verre, très pâle, sur toute la
     largeur ; puis une portion allumée, de largeur et de place tirées ; puis
     rien du tout un étage sur deux. Ce sont les portions qui font les
     bureaux encore occupés, et leur irrégularité qui fait la façade.

     Les étages sont comptés depuis le socle et non depuis le sommet de la
     barre. Deux tours voisines ont alors leurs bandeaux sur les mêmes
     lignes, comme deux immeubles d'une même rue ; comptés depuis le sommet,
     chaque tour avait sa propre grille et la file lisait comme un bruit.

     Le rang de l'étage entre dans le tirage, donc un étage allumé reste
     allumé quand la barre monte : la tour ne se rallume pas à chaque pas. */
  const inset = 2.5, ww = bw - inset * 2, gh = 1.4;
  ctx.fillStyle = lit;
  for (let y = base - FLOOR - 1; y > base - shoulder + 3; y -= FLOOR) {
    const r = Math.round((base - y) / FLOOR);
    ctx.globalAlpha = a * (NIGHT ? 0.24 : 0.16);
    ctx.fillRect(x0 + inset, y, ww, gh);
    if (hash(idx, r, 1) % 100 < 46) {
      const w2 = ww * (0.22 + (hash(idx, r, 2) % 58) / 100);
      const ox = x0 + inset + (ww - w2) * (hash(idx, r, 3) % 100) / 100;
      /* La nuit, une fenêtre allumée est allumée : elle ne se nuance pas.
         Le jour, un reflet de ciel n'a pas de raison d'être à pleine force,
         et c'est là que la nuance sert. */
      ctx.globalAlpha = a * (NIGHT ? 0.76 + (hash(idx, r, 4) % 24) / 100
                                   : 0.46 + (hash(idx, r, 4) % 44) / 100);
      ctx.fillRect(ox, y, w2, gh);
    }
  }
  ctx.globalAlpha = a;
}

/* L'horizon : une file de tours fixes derrière le ruban.

   Elles ne portent aucune donnée — c'est de la profondeur, pas une mesure —
   donc elles ne glissent pas avec le pas, n'ont pas de fenêtres et restent
   dans la couleur des filets. Ce sont elles, et non les barres, qui portent
   les châteaux d'eau, les antennes et les couronnements : au-dessus d'une
   barre, le moindre de ces objets mentirait sur la hauteur, qui est la
   valeur. Ici, rien à mentir.

   Sans cette file, les barres se tiennent sur du vide. Avec elle, le ruban
   est une rue. */
function skyline(w, base, top) {
  ctx.globalAlpha = NIGHT ? 0.55 : 0.8;
  ctx.fillStyle = C.rule;
  const unit = 13, room = (base - top) * 0.3;
  for (let i = 0, n = Math.ceil(w / unit); i <= n; i++) {
    const hh = 9 + (hash(i, 5, 11) % 100) / 100 * room;
    const x = i * unit, bw = unit - 3, y = base - hh;
    ctx.fillRect(x, y, bw, hh);
    const k = hash(i, 7, 3) % 5;
    if (k === 0) {
      ctx.fillRect(x + Math.round(bw * 0.4), y - 7, 1, 7);
    } else if (k === 1) {
      ctx.fillRect(x + 1, y - 4, Math.round(bw * 0.5), 4);
    } else if (k === 2) {
      ctx.fillRect(x + Math.round(bw * 0.2), y - 3, Math.round(bw * 0.6), 3);
    }
  }
  ctx.globalAlpha = 1;
}

/* La liasse : des bandes de hauteur fixe, empilées depuis le sommet. La
   hauteur du billet est en pixels et non en dollars — à mille cinq cents
   dollars le billet, une heure de pointe en aurait empilé cent cinquante,
   et personne ne compte cent cinquante bandes. C'est une texture qui dit
   « de l'argent », pas un compte qu'on inviterait à vérifier.

   Le décalage alterné d'un pixel fait la liasse mal rangée. Sans lui, les
   bandes régulières se lisent comme un dégradé et non comme des billets. */
const BILL_H = 4, BILL_GAP = 2;

function bills(x0, base, bw, bh, idx, col) {
  ctx.fillStyle = col;
  const pitch = BILL_H + BILL_GAP;
  let i = 0;
  /* Du haut vers le bas : c'est le billet du sommet qui porte la valeur, et
     lui seul doit tomber pile sur `bh`.

     Le dernier billet est étiré jusqu'au socle. Empilés du haut avec un pas
     fixe, les billets laissaient sinon le reste de la division en vide sous
     la liasse — jusqu'à cinq pixels de barre flottante, et le ruban se lit
     justement à ras du socle. */
  for (let top = bh; top > 0; top -= pitch, i++) {
    const next = top - pitch;
    const hh = next > 0 ? Math.min(BILL_H, top) : top;
    ctx.fillRect(x0 + (hash(idx, i, 7) % 3) - 1, base - top, bw, hh);
  }
}

const PAINT = { trips: tower, revenue: bills };

/* Les teintes de la barre : le pas courant à l'encre, les autres dans le
   gris de fond.

   Le pas courant était rouge. Le rouge est maintenant pris, et deux fois :
   l'aiguille du cadran et les chiffres du taximètre. Trois rouges dans un
   même tableau de bord se disputent le regard, et celui du ruban était le
   moins utile des trois — le pas courant est déjà sur la ligne du présent,
   déjà le seul chiffré, déjà le seul à pleine opacité. L'encre suffit. */
function paintBar(x0, base, bw, bh, idx, here) {
  (PAINT[S.measure] || tower)(x0, base, bw, bh, idx,
    here ? MASS_HERE : MASS_FAR, GLASS);
}

function draw() {
  const w = cv.clientWidth, h = cv.clientHeight;
  ctx.clearRect(0, 0, w, h);
  const { vals, max } = series();
  const n = vals.length;
  const t = phase();
  const { span, behind: BEHIND, ahead: AHEAD } = window_();
  const pitch = w / span;
  const bw = Math.min(pitch * 0.58, 26);
  /* `top` garde au-dessus des barres de quoi écrire la valeur du pas
     courant, et laisse passer le titre posé en surimpression : sans cette
     réserve, le nombre sort du canvas ou se superpose au titre. */
  const base = h - 34, top = 34 + HEAD;
  skyline(w, base, top);

  const scale = META.cycles[S.scope].scales[S.measure];
  ctx.textAlign = "center";

  for (let k = -BEHIND; k <= AHEAD; k++) {
    const idx = ((S.step + k) % n + n) % n;
    /* La position retire la phase : entre deux pas, tout le ruban glisse
       d'une fraction de case vers la gauche. */
    const x = (NOW_X * span + k - t) * pitch;
    if (x < -pitch || x > w + pitch) { continue; }
    const v = vals[idx] / max;
    const bh = Math.max(2, v * (base - top));
    const here = k === 0;
    /* Les barres loin du présent s'effacent : le ruban n'a pas de bord net,
       il se perd des deux côtés. */
    const fade = Math.max(0, 1 - Math.abs(k - (here ? 0 : 0)) / (span * 0.62));
    ctx.globalAlpha = here ? 1 : 0.3 + fade * 0.58;
    paintBar(x - bw / 2, base, bw, bh, idx, here);

    /* Une graduation sur deux sur les cycles dont le pas est une heure : la
       journée en pose vingt-quatre et la semaine cent soixante-huit, et à
       onze pixels de corps les nombres se touchaient. Les heures paires
       suffisent à situer, et le pas courant garde la sienne quoi qu'il
       arrive — c'est celui qu'on cherche. L'année n'a que douze pas, elle
       les garde tous. */
    const hourly = S.scope === "hour";
    if (here || !hourly || idx % 2 === 0) {
      ctx.globalAlpha = here ? 0.95 : 0.3 + fade * 0.4;
      ctx.fillStyle = here ? C.ink : C.dim;
      ctx.font = (here ? "600 " : "") + "11px system-ui, sans-serif";
      ctx.fillText(META.cycles[S.scope].short[idx], x, base + 16);
    }
    if (here) {
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillStyle = C.ink;
      ctx.fillText(fmt(vals[idx] * scale, S.measure), x, base - bh - 7);
    }
  }
  ctx.globalAlpha = 1;

  /* Le repère du présent : un trait vertical discret, sous les barres. */
  const nx = NOW_X * w;
  ctx.strokeStyle = C.rule;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(nx, top - 6); ctx.lineTo(nx, base + 4);
  ctx.stroke();
}
</script>
"""

# ---------------------------------------------------------------------------
# Composant 4 — les flux entre arrondissements
# ---------------------------------------------------------------------------
_FLOWS = r"""
<svg id="svg" style="position:absolute; inset:0; width:100%; height:100%;"></svg>
<script>
const SVG = document.getElementById("svg");
const NS = "http://www.w3.org/2000/svg";
/* Les flux internes à un secteur sont écartés : ils ne décrivent aucun
   déplacement à cette échelle, et le plus gros d'entre eux écraserait tout
   le reste en un seul ruban. Le bloc parle des liaisons. */
let NODES = null, LINKS = null;

function boot() {
  layout();
  window.addEventListener("resize", () => { layout(); });
  frame();
}

function flowRows() { return META.flows[S.scope].filter(f => f.s !== f.t); }

/* Les positions sont calculées une fois, sur le total du cycle entier, et ne
   bougent plus. Un Sankey qui replace ses nœuds à chaque pas saute ; ici
   seule l'épaisseur des rubans change. */
function layout() {
  const rows = flowRows();
  const total = {};
  for (const f of rows) {
    const sum = f.v.reduce((a, b) => a + b, 0);
    total["s" + f.s] = (total["s" + f.s] || 0) + sum;
    total["t" + f.t] = (total["t" + f.t] || 0) + sum;
  }
  const left = META.areas.filter(b => total["s" + b]);
  const right = META.areas.filter(b => total["t" + b]);
  const w = SVG.clientWidth, h = SVG.clientHeight;
  const pad = 14, gap = 11, nodeW = 11;
  /* Le titre est en surimpression : les nœuds commencent sous lui, sinon le
     plus haut des deux côtés passe derrière le texte. */
  const padTop = pad + HEAD;
  const padBot = pad + LEGH;
  const place = (list, side) => {
    const sum = list.reduce((a, b) => a + total[side + b], 0) || 1;
    const usable = h - padTop - padBot - gap * (list.length - 1);
    let y = padTop;
    const out = {};
    for (const b of list) {
      const hh = Math.max(8, (total[side + b] / sum) * usable);
      out[b] = { y0: y, y1: y + hh, h: hh };
      y += hh + gap;
    }
    return out;
  };
  NODES = { left: place(left, "s"), right: place(right, "t"),
            w: w, h: h, nodeW: nodeW, pad: pad };
  LINKS = rows.map(f => ({ ...f, max: Math.max(...f.v) }));
  /* Chaque ruban garde sa place sur le bord du nœud : l'ordre est figé, les
     épaisseurs seules varient, donc rien ne se croise en cours de route. */
  const cursor = {};
  LINKS.sort((a, b) => META.areas.indexOf(a.t) - META.areas.indexOf(b.t));
  for (const l of LINKS) {
    l.so = cursor["s" + l.s] = (cursor["s" + l.s] || 0);
    l.to = cursor["t" + l.t] = (cursor["t" + l.t] || 0);
    const share = l.max;
    cursor["s" + l.s] += share;
    cursor["t" + l.t] += share;
  }
  const share = {};
  for (const l of LINKS) {
    share["s" + l.s] = cursor["s" + l.s];
    share["t" + l.t] = cursor["t" + l.t];
  }
  for (const l of LINKS) {
    l.sTotal = share["s" + l.s];
    l.tTotal = share["t" + l.t];
  }
  SVG.innerHTML = "";
  draw(true);
}

function hex(b) { return META.areaColors[b] || C.dim; }

function draw(rebuild) {
  if (!NODES) { return; }
  const t = phase();
  const n = stepsOf(S.scope);
  const i = S.step, j = (S.step + 1) % n;
  const w = NODES.w, nodeW = NODES.nodeW;
  const x0 = NODES.pad, x1 = w - NODES.pad - nodeW;

  if (rebuild) {
    for (const side of ["left", "right"]) {
      for (const b of Object.keys(NODES[side])) {
        const nd = NODES[side][b];
        const r = document.createElementNS(NS, "rect");
        r.setAttribute("x", side === "left" ? x0 : x1);
        r.setAttribute("y", nd.y0);
        r.setAttribute("width", nodeW);
        r.setAttribute("height", nd.h);
        r.setAttribute("fill", hex(b));
        r.setAttribute("rx", 1.5);
        SVG.appendChild(r);
        const tx = document.createElementNS(NS, "text");
        tx.setAttribute("x", side === "left" ? x0 + nodeW + 7 : x1 - 7);
        /* Un nœud trop court ne peut pas porter son libellé en son milieu
           sans mordre sur le voisin : il le pose au-dessus de son arête. */
        tx.setAttribute("y", nd.h < 15 ? nd.y0 - 3 : nd.y0 + nd.h / 2 + 4);
        tx.setAttribute("fill", C.ink);
        tx.setAttribute("font-size", "11");
        tx.setAttribute("text-anchor", side === "left" ? "start" : "end");
        /* Contour de la couleur du fond : le libellé passe au-dessus des
           rubans sans que rien ne soit masqué. */
        tx.setAttribute("stroke", C.bg);
        tx.setAttribute("stroke-width", "3");
        tx.setAttribute("paint-order", "stroke");
        tx.textContent = b;
        SVG.appendChild(tx);
      }
    }
    for (const l of LINKS) {
      l.path = document.createElementNS(NS, "path");
      l.path.setAttribute("fill", "none");
      l.path.setAttribute("stroke", hex(l.s));
      l.path.setAttribute("stroke-opacity", "0.34");
      l.path.onmousemove = e => tipAt(e.clientX, e.clientY,
        "<b>" + l.s + " → " + l.t + "</b><br>"
        + FR(l.v[S.step]) + " courses");
      l.path.onmouseleave = () => tipAt(0, 0, null);
      SVG.appendChild(l.path);
    }
  }

  for (const l of LINKS) {
    const v = l.v[i] + (l.v[j] - l.v[i]) * t;
    const a = NODES.left[l.s], b = NODES.right[l.t];
    if (!a || !b) { continue; }
    const sw = Math.max(0.6, (v / l.sTotal) * a.h);
    const ay = a.y0 + (l.so / l.sTotal) * a.h + sw / 2;
    const by = b.y0 + (l.to / l.tTotal) * b.h
             + Math.max(0.6, (v / l.tTotal) * b.h) / 2;
    const mx = (x0 + nodeW + x1) / 2;
    l.path.setAttribute("stroke-width", sw);
    l.path.setAttribute("d",
      "M" + (x0 + nodeW) + "," + ay +
      "C" + mx + "," + ay + " " + mx + "," + by + " " + x1 + "," + by);
  }
}

function onState() { layout(); }
function frame() { if (META) { draw(false); } requestAnimationFrame(frame); }
</script>
"""
