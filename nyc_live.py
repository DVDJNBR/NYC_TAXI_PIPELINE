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

    hour    24 pas, minuit -> minuit, tous jours confondus
    dow      7 pas, lundi -> dimanche
    month   12 pas, janvier -> décembre

Les valeurs sont des moyennes par occurrence du pas, calculées en amont par
scripts/J_build_cycles.py — un total brut ferait chuter novembre de moitié
pour la seule raison qu'il n'est observé qu'une fois sur la période.

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
import streamlit.components.v1 as components

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

MEASURES = ["trips", "revenue", "distance"]
COLUMNS = {"trips": "TOTAL_TRIPS", "revenue": "TOTAL_REVENUE",
           "distance": "AVG_DISTANCE"}

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

# Durée d'un pas, en millisecondes. Une seconde par heure : le temps de
# parcourir la carte, de lire le pic qui monte et de revenir au ruban avant
# que le pas ne change. En deçà, la page bouge plus vite qu'on ne la lit.
#
# Un cycle court a besoin d'un pas plus long : sept jours à une seconde
# feraient une boucle de sept secondes, soit un battement plutôt qu'une
# lecture. La boucle vise la quarantaine de secondes dans les trois cas.
#
# Le pas tient maintenant une seconde et demie au moins : quatre nombres, une
# carte et un ruban changent ensemble, et il faut le temps d'en lire un avant
# que le suivant ne parte. Qui veut aller plus vite clique la barre voulue —
# le ruban est devenu une règle, pas seulement un affichage.
CYCLES = {
    "hour": {"steps": 24, "ms": 1700, "label": "Heure",
             "title": "Une journée type",
             "labels": [f"{h}h" for h in range(24)],
             "short": [f"{h}h" for h in range(24)]},
    "dow": {"steps": 7, "ms": 2800, "label": "Jour",
            "title": "Une semaine type", "labels": DOW_LABELS,
            "short": DOW_SHORT},
    "month": {"steps": 12, "ms": 2400, "label": "Mois",
              "title": "Une année type", "labels": MONTH_LABELS,
              "short": MONTH_SHORT},
}

# Zones trop rares pour porter une colonne : moins d'une course par jour en
# moyenne sur la période. Les garder allongerait le binaire sans rien ajouter.
MIN_TRIPS_PER_DAY = 1.0


def available() -> bool:
    return GEO.exists() and ZONE_CYCLE.exists() and OD_CYCLE.exists()


# Numéro de forme du payload. Il entre dans la clé du cache : sans lui, un
# changement de structure ne serait pas vu tant que les agrégats ne sont pas
# reconstruits, et le composant lirait un fichier à l'ancien format.
PAYLOAD_VERSION = 4


@st.cache_data(show_spinner=False)
def _build(version: int, geo_mtime: float, zone_mtime: float,
           od_mtime: float) -> dict:
    """Écrit les fichiers statiques et rend un résumé.

    Les dates de modification sont des arguments et non des lectures
    internes : elles font la clé du cache, qui s'invalide donc tout seul
    quand les agrégats sont reconstruits.
    """
    geo = {z["id"]: z for z in json.loads(GEO.read_text())}
    zc = pd.read_parquet(ZONE_CYCLE)
    od = pd.read_parquet(OD_CYCLE)

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
        "flows": _flows(od),
        "areas": nyc_zones.AREAS,
        "areaColors": nyc_zones.AREA_COLORS,
    }
    (STATIC / "nyc_live.json").write_text(json.dumps(meta, separators=(",", ":")))
    return {"zones": len(kept),
            "bin_ko": (STATIC / "nyc_live.bin").stat().st_size // 1024,
            "json_ko": (STATIC / "nyc_live.json").stat().st_size // 1024}


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
                  ZONE_CYCLE.stat().st_mtime, OD_CYCLE.stat().st_mtime)


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
    components.html(
        _html(_controls_body(), _cfg(scope=scope, measure=measure)),
        height=height, scrolling=False,
    )


def kpis(height: int = 160) -> None:
    """Quatre nombres qui suivent le pas de l'horloge."""
    meta()
    components.html(_html(_KPIS, _cfg()), height=height, scrolling=False)


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
    components.html(_html(_MAP, _cfg(view=VIEW, head=head)),
                    height=height, scrolling=False)


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
    components.html(_html(_CITY, _cfg(head=head, legend=legend)),
                    height=height, scrolling=False)


def profile(height: int = 240, title: str = "", hint: str = "") -> None:
    """Le ruban de barres qui défile vers la gauche."""
    meta()
    components.html(_html(_PROFILE, _cfg(head=_head(title, hint))),
                    height=height, scrolling=False)


def flows(height: int = 300, title: str = "", hint: str = "",
          legend: bool = False) -> None:
    """Les flux entre arrondissements, nœuds figés, rubans animés."""
    meta()
    components.html(
        _html(_FLOWS, _cfg(head=_head(title, hint), legend=legend)),
        height=height, scrolling=False)


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
(function () {
  if (!CFG.head || !CFG.head.title) { HEAD_EL.remove(); return; }
  HEAD_EL.innerHTML = "<b>" + CFG.head.title + "</b>"
              + (CFG.head.hint ? "<i>" + CFG.head.hint + "</i>" : "");
  HEAD_EL.querySelector("b").style.color = CFG.head.ink || C.ink;
  const hint = HEAD_EL.querySelector("i");
  if (hint) { hint.style.color = CFG.head.dim || C.dim; }
  const measure = () => { HEAD = HEAD_EL.offsetHeight + 8; };
  measure();
  window.addEventListener("resize", measure);
})();

/* Pendant de HEAD, en bas. La légende se replie sur une, deux ou trois
   lignes selon la largeur du bloc, donc sa hauteur se mesure. Pas `FOOT` :
   la maquette de ville nomme déjà ainsi l'empreinte au sol de ses tours, et
   deux déclarations du même nom au premier niveau du script faisaient
   tomber le composant entier. */
let LEGH = 0;
const LEG_EL = document.getElementById("legend");
function drawLegend() {
  if (!CFG.legend || !META.areas) { LEG_EL.remove(); return; }
  LEG_EL.innerHTML = META.areas.map(n =>
    '<span><i style="background:' + (META.areaColors[n] || C.dim)
    + '"></i>' + n + "</span>").join("");
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
  if (measure === "distance") { return v.toFixed(1).replace(".", ",") + " mi"; }
  return FR(v) + " courses";
}
const MEASURE_LABEL = { trips: "Courses", revenue: "Revenus",
                        distance: "Distance" };

/* Ordre d'affichage des trois mesures, découplé de META.measures — qui est
   l'ordre physique du binaire et ne peut pas bouger sans le réécrire.
   Courses passe en dernier : c'est la mesure par défaut, mais le ruban juste
   au-dessus la montre déjà barre par barre, donc elle n'a pas besoin de la
   place où l'œil arrive d'abord. */
const MEASURE_ORDER = ["revenue", "distance", "trips"];
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
const SERIES = {}, TOPS = {}, LOWS = {};

/* État partagé. `at` est l'heure epoch du dernier changement de pas : chaque
   composant en déduit sa phase et anime en continu sans recevoir une image
   par trame. Date.now() et non performance.now(), dont l'origine diffère
   d'une iframe à l'autre. */
let S = { scope: "hour", measure: "trips", step: 0, playing: true,
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
    SERIES[scope] = {}; TOPS[scope] = {}; LOWS[scope] = {};
    const n = META.cycles[scope].steps;
    for (const m of META.measures) {
      const a = new Uint16Array(BIN, offset, Z * n);
      offset += Z * n * 2;
      SERIES[scope][m] = a;
      /* Le maximum, et le plus petit relevé non nul. Les deux servent à la
         carte : rapportée à zéro, la distance moyenne — qui ne descend
         jamais sous deux milles — mettait toutes les zones au plafond et la
         ville devenait un bloc doré. Le zéro, lui, reste le zéro : c'est une
         zone sans course, pas une zone à courses courtes. */
      let hi = 1, lo = 65535;
      for (let i = 0; i < a.length; i++) {
        if (a[i] > hi) { hi = a[i]; }
        if (a[i] > 0 && a[i] < lo) { lo = a[i]; }
      }
      TOPS[scope][m] = hi;
      LOWS[scope][m] = lo < hi ? lo : 0;
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

    Les trois groupes pendent d'un seul portique et non d'un portique chacun :
    c'est un assemblage qu'on voit sur une voie rapide, pas trois poteaux
    plantés côte à côte. L'armature est un treillis — deux membrures et leurs
    diagonales croisées — parce qu'une barre lisse fait du plastique, et elle
    passe derrière le bord haut des plaques, visible dans les intervalles,
    avec les montants qui descendent aux deux bouts.

    Le pas courant est du mobilier lui aussi, mais pas un panneau : ce n'est
    pas une destination. C'est une borne, posée entre les voies, au bas du
    rail — un fût à sommet arrondi, bande de couleur en tête, chiffre en
    noir. Les plaques pendent du portique, la borne tient au sol.

    Les six plaques ont une seule taille, le mot centré dessus : une travée
    se commande en série, et six largeurs au gabarit du texte faisaient six
    objets là où il n'y en a qu'un, répété.

    Les flèches sont celles d'une affectation de voies : bas-gauche, bas,
    bas-droite, posées sous le mot et non à côté. Elles disent que les trois
    plaques d'un même groupe sont trois sorties d'un même embranchement.

    De nuit, pas de portique : un panneau rétroréfléchissant sans phare pour
    l'éclairer n'est qu'une plaque grise. La ligne garde alors le cerclage
    chromé et le faisceau ambré du phare, et le clignotant au survol.
    """
    m = nyc_theme.materials()
    dark = m["mode"] == "dark"
    p = nyc_theme.palette()
    accent = p["ACCENT"]

    # Montants du portique : le même métal que les membrures, pris dans
    # l'autre sens. Un dégradé vertical sur une barre verticale ferait un
    # tube uni.
    post = ("linear-gradient(to right, #86868f 0%, #cbcbd3 44%,"
            " #a3a3ac 56%, #767680 100%)")

    # Le treillis : deux nappes de diagonales en sens inverse, serrées entre
    # les membrures. L'une plus claire que l'autre, parce qu'une diagonale qui
    # monte et une qui descend ne prennent pas la même lumière — à valeur
    # égale, le croisillon se lirait comme une trame imprimée.
    def lattice(angle, colour):
        return (f"repeating-linear-gradient({angle}deg,"
                f" transparent 0 5.5px, {colour} 5.5px 7.5px,"
                " transparent 7.5px 13px)")

    chord = m["metal"]
    truss = f"{chord}, {chord}, {lattice(56, '#c2c2ca')}, {lattice(-56, '#9a9aa4')}"

    gantry = "" if dark else f"""
  #rail {{ align-items: flex-start; }}
  /* Deux portiques de trois plaques, pas une travée qui traverse tout : ce
     qu'on mesure et le pas du cycle sont deux embranchements distincts, et
     une poutre continue entre les deux en aurait fait un seul.

     Chaque groupe débord de ses plaques : l'écart latéral laisse voir le
     bout de la poutre et le montant qui y descend. Sans lui, l'armature
     était un objet dont on ne voyait jamais les extrémités.

     La poutre passe à mi-hauteur des plaques. C'est là qu'elle est sur une
     vraie travée : les panneaux sont boulonnés devant elle, pas pendus
     dessous, et c'est le bandeau visible dans les intervalles qui dit
     qu'elle continue derrière. */
  .set {{ padding: 0 15px; }}
  .set::before {{
      content: ""; position: absolute; left: 0; right: 0; top: 16px;
      height: 13px;
      background-image: {truss};
      background-size: 100% 3px, 100% 3px, 100% 7px, 100% 7px;
      background-position: top left, bottom left, left 3px, left 3px;
      background-repeat: repeat-x;
      opacity: 0.9;
      z-index: 0; }}
  /* Les montants descendent des bouts de la poutre et passent plus bas que
     les plaques : un portique tient au sol, pas dans l'air. */
  .set::after {{
      content: ""; position: absolute; left: 0; right: 0;
      top: 18px; bottom: -7px;
      background-image: {post}, {post};
      background-size: 5px 100%, 5px 100%;
      background-position: left top, right top;
      background-repeat: no-repeat;
      opacity: 0.9;
      z-index: 0; }}
  /* Une travée se commande en une seule taille : six plaques identiques,
     le mot centré dessus. Des largeurs au gabarit du texte faisaient six
     objets différents là où il n'y a qu'une série. */
  .sign {{ box-sizing: border-box; width: 92px; text-align: center;
          padding: 9px 4px 22px; }}
"""

    # De jour la plaque porte sa propre matière : joncs chromés sertis en haut
    # et en bas, tôle peinte entre les deux. De nuit il n'y a pas de plaque,
    # mais un phare : le faisceau sous le mot et le cerclage qui le projette.
    #
    # Les joncs sont la pièce, pas la décoration de l'actif : deux panneaux
    # d'un même portique sont sertis pareil, et seule la tôle entre les joncs
    # change de couleur.
    WHITE_FACE = ("linear-gradient(152deg, #ffffff 0%, #f6f6f4 46%,"
                  " #ececeb 100%)")

    def plate(face):
        if dark:
            return ""
        return (f"background-image: {m['metal_band']}, {m['metal_band']},"
                f" {face};"
                " background-size: 100% 3px, 100% 3px, 100% 100%;"
                " background-position: top left, bottom left, center;"
                " background-repeat: no-repeat;"
                " border-color: transparent;")

    on_face = plate(m["sign_face"])
    on_ink = accent if dark else m["sign_ink"]
    off_face = plate(WHITE_FACE)
    off_ink = p["TEXT_DIM"] if dark else "#15151a"

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

    # Les flèches d'affectation de voie, sous le mot et non à côté : elles
    # disent que les trois plaques d'un groupe sont trois sorties d'un même
    # embranchement. Bas-gauche, bas, bas-droite, dans l'ordre d'affichage.
    #
    # Leur place est tenue dès le repos, sinon six boutons se décalent au
    # passage de la souris. Elles descendent de quelques pixels en
    # apparaissant : une flèche de voie se lit de haut en bas.
    #
    # Un seul masque pour les trois : le glyphe pointe à droite, la rotation
    # fait le reste. 45° pour bas-droite, 90° pour bas, 135° pour bas-gauche —
    # une rotation CSS tourne dans le sens des aiguilles.
    TURN = {"i0": 135, "i1": 90, "i2": 45}
    lanes = "".join(f"""
  .sign.{k}::after {{ transform: translateY(-5px) rotate({a}deg); }}
  .sign.{k}:hover::after, .sign.{k}.on::after {{
      transform: translateY(0) rotate({a}deg); }}
  @media (prefers-reduced-motion: reduce) {{
      .sign.{k}::after {{ transform: rotate({a}deg); }} }}"""
                    for k, a in TURN.items())

    arrow = "" if dark else f"""
  .sign::after {{
      content: ""; position: absolute;
      left: 50%; bottom: 7px; width: 15px; height: 15px; margin-left: -7.5px;
      background-color: currentColor;
      -webkit-mask: {m['arrow']};
      mask: {m['arrow']};
      opacity: 0;
      transition: opacity 0.16s ease, transform 0.26s cubic-bezier(0.22,0.7,0.3,1); }}
  .sign:hover::after, .sign.on::after {{ opacity: 1; }}
  {lanes}
  @media (prefers-reduced-motion: reduce) {{
      .sign::after {{ transition: opacity 0.16s ease; }} }}
"""

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
  {clock}
  {arrow}
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
let timer = null;

/* Le rang sert à la flèche de voie : bas-gauche, bas, bas-droite. Il est
   porté par une classe et non par un style en ligne, pour que la règle de
   survol et celle du repos vivent au même endroit que le reste du panneau. */
function sign(label, active, i, onclick) {
  const b = document.createElement("button");
  b.className = "sign i" + i + (active ? " on" : "");
  b.textContent = label;
  b.onclick = onclick;
  return b;
}

function paintButtons() {
  const sc = document.getElementById("scopes");
  sc.innerHTML = "";
  Object.keys(META.cycles).forEach((k, i) => {
    sc.appendChild(sign(META.cycles[k].label, S.scope === k, i, () => {
      S.scope = k; S.step = 0; S.at = Date.now(); publish();
    }));
  });
  const me = document.getElementById("measures");
  me.innerHTML = "";
  measuresInOrder().forEach((m, i) => {
    me.appendChild(sign(MEASURE_LABEL[m], S.measure === m, i, () => {
      S.measure = m; publish();
    }));
  });
  /* Le pas courant est écrit en clair : le ruban le montre aussi, mais le
     lire ici évite de chercher quelle barre porte l'accent.
     En abrégé, parce qu'une borne porte une marque courte et non une
     phrase — « 18H », « MAR », « JUIL ». Le cycle actif est lu juste à
     côté, donc rien n'est ambigu. */
  document.getElementById("now").textContent =
    META.cycles[S.scope].short[S.step];
  document.getElementById("play").textContent = S.playing ? "❚❚" : "▶";
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
# Composant 1 bis — les quatre nombres du pas courant
# ---------------------------------------------------------------------------
_KPIS = r"""
<style>
  #grid { position: absolute; inset: 0; display: grid;
          grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
          gap: 8px 14px; }
  .kpi { display: flex; flex-direction: column; justify-content: center;
         min-width: 0; }
  .k-lab { font-size: 0.64rem; letter-spacing: 0.09em;
           text-transform: uppercase; }
  .k-val { font-size: 1.15rem; font-weight: 600; line-height: 1.25;
           font-variant-numeric: tabular-nums; }
  .k-sub { font-size: 0.68rem; line-height: 1.3; }
  /* Chaque valeur est une case où deux textes se superposent, le sortant et
     l'entrant : ils occupent la même ligne et se croisent sur place. Un seul
     élément qu'on vide puis remplit laisserait une case vide entre les deux,
     et c'est ce trou qui se voit, pas le changement. */
  .k-lab, .k-val, .k-sub { display: grid; min-width: 0; }
  .k-lab > span, .k-val > span, .k-sub > span {
      grid-area: 1 / 1; min-width: 0;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      transition: opacity 0.22s ease, transform 0.22s ease; }
  /* Le sortant monte, l'entrant vient d'en dessous : le même sens que le
     ruban, où le passé file et l'à-venir arrive. */
  .k-lab > span.out, .k-val > span.out, .k-sub > span.out {
      opacity: 0; transform: translateY(-0.3em); }
  .k-lab > span.in, .k-val > span.in, .k-sub > span.in {
      opacity: 0; transform: translateY(0.3em); }
  @media (prefers-reduced-motion: reduce) {
      .k-lab > span, .k-val > span, .k-sub > span {
          transition: opacity 0.22s ease; }
      .k-lab > span.out, .k-val > span.out, .k-sub > span.out,
      .k-lab > span.in, .k-val > span.in, .k-sub > span.in {
          transform: none; } }
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
/* Quatre nombres, une seule règle : ils doivent bouger dans le cycle. Un
   nombre qui ne bouge pas n'appartient pas au cockpit, il appartient à la
   fiche signalétique — et la fiche est maintenant en pied de page.

   Mesuré sur les 24 pas de l'heure :
     — le mile coûte 5,83 $ à 5 h et 9,37 $ à 18 h              (x1,6)
     — hors Manhattan, 9,4 % des courses à 18 h, 26,7 % à 5 h   (x2,8)
     — la moitié des courses tient dans 9 zones à 2 h, 17 à 5 h  (x1,9)

   Écarté : l'encaissé total, corrélé aux courses à 0,995, qui les redit en
   dollars ; la zone la plus active, qui est Midtown vingt-deux pas sur
   vingt-quatre ; le prix d'une course, qui ne varie que de 24,30 à 34,71 $
   et que le prix du mile dit mieux — un mile cher, c'est de la circulation,
   une course chère peut n'être qu'un long trajet.

   La quatrième case suit le panneau allumé. Sans elle, choisir « Distance »
   ne changeait rien de lisible hors de la carte : la commande pilotait un
   bloc et laissait les nombres où ils étaient. */
const CELLS = ["permile", "outer", "spread", "pick"];
const CELL_LABEL = { permile: "Le prix du mile",
                     outer: "Hors Manhattan",
                     spread: "La moitié des courses" };

/* Les zones qui ne sont pas de Manhattan, relevées une fois au démarrage. */
let OUTER = [];

function boot() {
  const g = document.getElementById("grid");
  for (const key of CELLS) {
    g.insertAdjacentHTML("beforeend",
      '<div class="kpi" id="c-' + key + '">'
      + '<div class="k-lab" style="color:' + C.dim + '"></div>'
      + '<div class="k-val"></div>'
      + '<div class="k-sub" style="color:' + C.dim + '"></div></div>');
  }
  OUTER = [];
  for (let z = 0; z < Z; z++) {
    if (META.zones[z].borough !== "Manhattan") { OUTER.push(z); }
  }
  onState();
}

/* Somme d'une mesure sur un sous-ensemble de zones, à un pas donné. */
function sumAt(step, measure, zones) {
  const n = stepsOf(S.scope);
  const idx = ((step % n) + n) % n;
  const a = SERIES[S.scope][measure];
  let s = 0;
  for (const z of zones) { s += a[z * n + idx]; }
  return s * META.cycles[S.scope].scales[measure];
}

/* Milles parcourus par la ville : distance moyenne de chaque zone multipliée
   par ses courses. Sommer les moyennes donnerait le même poids à une zone de
   trois courses qu'à Midtown. */
function cityMiles(step) {
  const n = stepsOf(S.scope);
  const idx = ((step % n) + n) % n;
  const d = SERIES[S.scope].distance, t = SERIES[S.scope].trips;
  const ds = META.cycles[S.scope].scales.distance;
  const ts = META.cycles[S.scope].scales.trips;
  let mi = 0;
  for (let z = 0; z < Z; z++) {
    mi += d[z * n + idx] * ds * t[z * n + idx] * ts;
  }
  return mi;
}

/* Ce que coûte un mille parcouru. C'est le seul nombre du bloc qui parle de
   circulation : le tarif compte au mille et à la minute, donc un mille cher
   est un mille lent. */
function perMile(step) {
  const mi = cityMiles(step);
  return mi ? cityAt(step, "revenue") / mi : 0;
}

/* Part des courses qui ne partent pas de Manhattan. La nuit la ville se
   replie sur l'île ; au petit matin, un quart des départs sont ailleurs. Le
   ruban dit quelles liaisons, ce nombre dit combien. */
function outerShare(step) {
  const t = cityAt(step, "trips");
  return t ? sumAt(step, "trips", OUTER) / t * 100 : 0;
}

/* Combien de zones il faut pour tenir la moitié des courses. La ville se
   resserre la nuit sur une poignée de quartiers et s'étale au petit matin ;
   un seul nombre le dit, là où une carte demande d'être lue. */
function halfSpread(step) {
  const n = stepsOf(S.scope);
  const idx = ((step % n) + n) % n;
  const a = SERIES[S.scope].trips;
  const v = [];
  let tot = 0;
  for (let z = 0; z < Z; z++) {
    const x = a[z * n + idx];
    if (x > 0) { v.push(x); tot += x; }
  }
  v.sort((x, y) => y - x);
  let acc = 0, k = 0;
  while (k < v.length && acc < tot / 2) { acc += v[k]; k++; }
  return k;
}

function cellValue(key, step) {
  if (key === "permile") { return perMile(step); }
  if (key === "outer") { return outerShare(step); }
  if (key === "spread") { return halfSpread(step); }
  return cityValue(step, S.measure);
}

function cellLabel(key) {
  return key === "pick" ? MEASURE_LABEL[S.measure] : CELL_LABEL[key];
}

function cellFormat(key) {
  if (key === "permile") {
    return x => "$" + x.toFixed(2).replace(".", ",") + " / mi";
  }
  if (key === "outer") {
    return x => x.toFixed(1).replace(".", ",") + " %";
  }
  if (key === "spread") { return x => Math.round(x) + " zones"; }
  return x => fmt(x, S.measure);
}

/* La courbe du cycle entier, mise en cache : elle ne change qu'avec le cycle
   et, pour la dernière case, avec la mesure choisie. */
const SPARK = {};
function cellSeries(key) {
  const ck = S.scope + "/" + key + (key === "pick" ? "/" + S.measure : "");
  if (!SPARK[ck]) {
    const n = stepsOf(S.scope);
    const out = new Array(n);
    for (let i = 0; i < n; i++) { out[i] = cellValue(key, i); }
    SPARK[ck] = out;
  }
  return SPARK[ck];
}


/* Somme d'une mesure sur toutes les zones, à un pas donné. */
function cityAt(step, measure) {
  const n = stepsOf(S.scope);
  const idx = ((step % n) + n) % n;
  const a = SERIES[S.scope][measure];
  let s = 0;
  for (let z = 0; z < Z; z++) { s += a[z * n + idx]; }
  return s * META.cycles[S.scope].scales[measure];
}

/* Distance moyenne de la ville : une moyenne pondérée par les courses, pas
   une moyenne des moyennes de zones — sans quoi une zone à trois courses
   compterait autant que Midtown. */
function cityDistance(step) {
  const n = stepsOf(S.scope);
  const idx = ((step % n) + n) % n;
  const d = SERIES[S.scope].distance, t = SERIES[S.scope].trips;
  const ds = META.cycles[S.scope].scales.distance;
  let num = 0, den = 0;
  for (let z = 0; z < Z; z++) {
    const trips = t[z * n + idx];
    num += d[z * n + idx] * ds * trips;
    den += trips;
  }
  return den ? num / den : 0;
}

function cityValue(step, measure) {
  if (measure === "distance") { return cityDistance(step); }
  return cityAt(step, measure);
}

/* Moyenne de la case sur le cycle entier : c'est elle qui donne son sens au
   nombre du pas. « 2 273 courses » ne dit rien ; « +38 % sur la journée »
   situe l'heure dans la journée. Elle se lit sur la courbe déjà calculée. */
function cycleAverage(key) {
  const v = cellSeries(key);
  let s = 0;
  for (const x of v) { s += x; }
  return v.length ? s / v.length : 0;
}

const ROLL_MS = 420, STAGGER = 70;
const SOFT = window.matchMedia
  && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Dernière valeur atteinte par case, et animation en cours : sans la
   première, le compteur repartirait de l'arrivée précédente affichée, qui
   peut être un état intermédiaire si le pas change pendant la course. */
const LAST = {}, ANIM = {};

/* Compte de l'ancienne valeur à la nouvelle. `render` fait le texte, ce qui
   garde ici la même mise en forme qu'ailleurs — francs, miles, pourcentage —
   sans que cette fonction ait à savoir laquelle. */
function rollCell(el, key, to, render, delay) {
  let span = el.firstElementChild;
  if (!span) { span = document.createElement("span"); el.appendChild(span); }
  const from = LAST[key] === undefined ? to : LAST[key];
  LAST[key] = to;
  if (ANIM[key]) { cancelAnimationFrame(ANIM[key]); ANIM[key] = 0; }
  if (from === to || SOFT) { span.textContent = render(to); return; }
  span.textContent = render(from);
  const t0 = performance.now() + delay;
  const tick = now => {
    /* Avant t0 la fraction est nulle : c'est ainsi que se fait l'attente de
       la case, sans minuteur à annuler si le pas change entre-temps. */
    const u = Math.max(0, Math.min(1, (now - t0) / ROLL_MS));
    const e = 1 - Math.pow(1 - u, 3);
    span.textContent = render(from + (to - from) * e);
    ANIM[key] = u < 1 ? requestAnimationFrame(tick) : 0;
  };
  ANIM[key] = requestAnimationFrame(tick);
}

/* Remplace le texte d'une case en croisant l'ancien et le nouveau. Rien ne
   bouge si le texte est le même : la distance moyenne et la zone la plus
   active tiennent souvent plusieurs pas, et les faire clignoter sans qu'elles
   changent dirait un changement qui n'a pas eu lieu. */
function setCell(el, text) {
  const cur = el.lastElementChild;
  if (cur && cur.textContent === text) { return; }
  const next = document.createElement("span");
  next.textContent = text;
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

function signed(ratio) {
  const pct = Math.round((ratio - 1) * 100);
  return (pct > 0 ? "+" : "") + pct + " %";
}

function onState() {
  CELLS.forEach((key, rank) => {
    const cell = document.getElementById("c-" + key);
    const avg = cycleAverage(key);
    const render = cellFormat(key);
    const delay = rank * STAGGER;
    /* La case liée au panneau allumé prend la brique, comme le panneau et
       comme la barre du présent dans le ruban : trois endroits, une seule
       couleur pour dire « c'est ceci que vous avez choisi ». */
    cell.querySelector(".k-val").style.color =
      key === "pick" ? C.brick : C.ink;
    setCell(cell.querySelector(".k-lab"), cellLabel(key));
    rollCell(cell.querySelector(".k-val"), key, cellValue(key, S.step),
             render, delay);
    rollCell(cell.querySelector(".k-sub"), key + "/sub",
             cellValue(key, S.step),
             x => (avg ? signed(x / avg) + " vs moyenne" : ""), delay);
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

/* Hauteur de la plus haute colonne, en mètres. Chaque mesure est ramenée à
   son propre maximum : sans quoi la distance moyenne resterait au sol. */
const TOP_M = 9000;

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
      tipAt(info.x, info.y, "<b>" + d.name + "</b><br>" + d.borough + " · "
            + fmt(scaled(d.i, S.step, S.scope, S.measure), S.measure));
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
    const hi = TOPS[S.scope][S.measure];
    const lo = LOWS[S.scope][S.measure];
    const span = Math.max(1, hi - lo);
    const rows = META.zones.map((zone, i) => {
      const a = value(i, S.step, S.scope, S.measure);
      const b = value(i, S.step + 1, S.scope, S.measure);
      const raw = a + (b - a) * t;
      return { ...zone, i: i, v: raw <= 0 ? 0 : Math.max(0, (raw - lo) / span) };
    });
    deckgl.setProps({ layers: [new deck.PolygonLayer({
      id: "zones", data: rows, extruded: true, filled: true, pickable: true,
      getPolygon: d => d.polygon,
      getElevation: d => Math.max(d.v * TOP_M, 30),
      /* Rampe en puissance : la racine remonte les zones moyennes, qui
         resteraient autrement indistinctes du fond. */
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
const REST = 70;              /* altitude des taxis posés, entre les tours  */
const SPEED = 0.055;          /* degrés par seconde : vitesse constante     */
const TAXI_PX = 6.2;

const PER_AXIS = 260;         /* courses par axe avant d'en ouvrir un autre */
const MAX_AXIS = 5;
const AXIS_GAP = 0.0062;
const JITTER = 0.0011;
const IN = 0.20, OUT = 0.80;  /* part du trajet passée à rejoindre l'axe    */
const KLAT = Math.cos(40.75 * Math.PI / 180);

const SECTORS = [0, 1, 2, 3, 4, 5, 6];

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
let MOV = null, SIT = null, AXES = [], STATIC = [];
let deckgl = null, VIEWSTATE = null, LABEL_POS = null, started = 0;

function pick(s) { return POOL[s][(Math.random() * POOL[s].length) | 0]; }

/* Les 24 pas de l'heure sont la seule granularité de ce payload. Sur les
   autres cycles on montre la journée moyenne plutôt que de laisser le bloc
   figé sur un pas qui ne veut rien dire. */
function matrix() {
  if (S.scope === "hour") { return CITY.trips[S.step % 24]; }
  const n = CITY.order.length;
  const out = [];
  for (let a = 0; a < n; a++) {
    const row = [];
    for (let b = 0; b < n; b++) {
      let v = 0;
      for (let h = 0; h < 24; h++) { v += CITY.trips[h][a][b]; }
      row.push(v / 24);
    }
    out.push(row);
  }
  return out;
}

function axesFor(a, b, n) {
  const A = CITY.lonlat[a], B = CITY.lonlat[b];
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

function build() {
  const m = matrix();
  const o = [], e1 = [], e2 = [], d = [], inv = [], ph = [];
  const rp = [];
  AXES = [];
  for (const a of SECTORS) {
    for (const b of SECTORS) {
      const n = Math.round(m[a][b]);
      if (n <= 0) { continue; }
      if (a === b) {
        for (let k = 0; k < n; k++) {
          const p = pick(a);
          rp.push(p[0], p[1], REST);
        }
        continue;
      }
      const na = Math.max(1, Math.min(MAX_AXIS, Math.round(n / PER_AXIS)));
      const ax = axesFor(a, b, na);
      for (const q of ax) {
        AXES.push({ s: [q[0][0], q[0][1], ALT], t: [q[1][0], q[1][1], ALT] });
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
  LABEL_POS = labelSpots();
  const nm = inv.length, nr = rp.length / 3;
  MOV = { n: nm, pos: new Float32Array(nm * 3), ang: new Float32Array(nm),
          o: new Float32Array(o), e1: new Float32Array(e1),
          e2: new Float32Array(e2), d: new Float32Array(d),
          inv: new Float32Array(inv), ph: new Float32Array(ph) };
  const ra = new Float32Array(nr);
  for (let i = 0; i < nr; i++) { ra[i] = Math.random() * 360; }
  SIT = { n: nr, pos: new Float32Array(rp), ang: ra };
  STATIC = staticLayers();
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
  for (const a of SECTORS) {
    const c = CITY.lonlat[a];
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
  frame();
}

function totals(step) {
  /* Somme sur toutes les zones : le ruban parle de la ville entière, la
     carte parle de chaque quartier. */
  let s = 0;
  const n = stepsOf(S.scope);
  const idx = ((step % n) + n) % n;
  const a = SERIES[S.scope][S.measure];
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
    ctx.fillStyle = here ? C.brick : C.muted;
    ctx.fillRect(x - bw / 2, base - bh, bw, bh);

    ctx.globalAlpha = here ? 0.95 : 0.3 + fade * 0.4;
    ctx.fillStyle = here ? C.ink : C.dim;
    ctx.font = (here ? "600 " : "") + "11px system-ui, sans-serif";
    ctx.fillText(META.cycles[S.scope].short[idx], x, base + 16);
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
