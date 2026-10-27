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

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import nyc_theme
import nyc_zones

ROOT = pathlib.Path(__file__).resolve().parent
STATIC = ROOT / "static"
GEO = ROOT / "data" / "reference" / "taxi_zones.json"
ZONE_CYCLE = ROOT / "data" / "aggregates" / "zone_cycle.parquet"
OD_CYCLE = ROOT / "data" / "aggregates" / "od_cycle.parquet"

CHANNEL = "nyc-cockpit"

MEASURES = ["trips", "revenue", "distance"]
COLUMNS = {"trips": "TOTAL_TRIPS", "revenue": "TOTAL_REVENUE",
           "distance": "AVG_DISTANCE"}

DOW_LABELS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi",
              "Samedi", "Dimanche"]
MONTH_LABELS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                "Juillet", "Août", "Septembre", "Octobre", "Novembre",
                "Décembre"]

# Durée d'un pas, en millisecondes. Un cycle court a besoin d'un pas plus
# long : sept jours à 500 ms feraient une boucle de trois secondes et demie,
# trop rapide pour qu'un œil suive la carte et le ruban en même temps.
CYCLES = {
    "hour": {"steps": 24, "ms": 520, "label": "Heure",
             "title": "Une journée type", "labels": [f"{h}h" for h in range(24)]},
    "dow": {"steps": 7, "ms": 900, "label": "Jour",
            "title": "Une semaine type", "labels": DOW_LABELS},
    "month": {"steps": 12, "ms": 760, "label": "Mois",
              "title": "Une année type", "labels": MONTH_LABELS},
}

# Zones trop rares pour porter une colonne : moins d'une course par jour en
# moyenne sur la période. Les garder allongerait le binaire sans rien ajouter.
MIN_TRIPS_PER_DAY = 1.0


def available() -> bool:
    return GEO.exists() and ZONE_CYCLE.exists() and OD_CYCLE.exists()


@st.cache_data(show_spinner=False)
def _build(geo_mtime: float, zone_mtime: float, od_mtime: float) -> dict:
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
             "polygon": _outer_ring(geo[z]["geometry"])}
            for z in kept
        ],
        "measures": MEASURES,
        "cycles": {k: {**v, "scales": scales[k]} for k, v in CYCLES.items()},
        "flows": _flows(od),
        "boroughs": nyc_zones.BOROUGHS,
        "boroughColors": nyc_zones.BOROUGH_COLORS,
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
    """Flux entre arrondissements, en clair : quelques milliers de nombres.

    Assez petit pour voyager en JSON. Les flux internes à un arrondissement
    sont gardés ici — c'est le composant qui décidera de les montrer ou non,
    et ils servent au total de chaque nœud.
    """
    out = {}
    valid = set(nyc_zones.BOROUGHS)
    od = od[od["PU_BOROUGH"].isin(valid) & od["DO_BOROUGH"].isin(valid)]
    for scope, cycle in CYCLES.items():
        part = od[od["SCOPE"] == scope]
        rows = []
        for (pu, do), g in part.groupby(["PU_BOROUGH", "DO_BOROUGH"]):
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
    return _build(GEO.stat().st_mtime, ZONE_CYCLE.stat().st_mtime,
                  OD_CYCLE.stat().st_mtime)


# ---------------------------------------------------------------------------
# Rendu des composants
# ---------------------------------------------------------------------------
def _cfg(**extra) -> str:
    p = nyc_theme.palette()
    return json.dumps({
        "channel": CHANNEL,
        "mode": nyc_theme.mode(),
        "colors": {
            "bg": p["BG_PAGE"], "ink": p["TEXT_LIGHT"], "dim": p["TEXT_DIM"],
            "rule": p["RULE"], "accent": p["ACCENT"], "panel": p["BG_PANEL"],
            "muted": p["BAR_MUTED"],
        },
        **extra,
    })


def _html(body: str, cfg: str) -> str:
    return _SHELL.replace("__CFG__", cfg).replace("__BODY__", body)


def controls(height: int = 88, scope: str = "hour",
             measure: str = "trips") -> None:
    """Les boutons-panneaux et l'horloge du cockpit."""
    meta()
    components.html(
        _html(_CONTROLS, _cfg(scope=scope, measure=measure)),
        height=height, scrolling=False,
    )


def city_map(height: int = 460, pitch: int = 52) -> None:
    """La carte à pics. Hauteur et couleur suivent la mesure du pas."""
    meta()
    components.html(_html(_MAP, _cfg(pitch=pitch)), height=height,
                    scrolling=False)


def profile(height: int = 240) -> None:
    """Le ruban de barres qui défile vers la gauche."""
    meta()
    components.html(_html(_PROFILE, _cfg()), height=height, scrolling=False)


def flows(height: int = 300) -> None:
    """Les flux entre arrondissements, nœuds figés, rubans animés."""
    meta()
    components.html(_html(_FLOWS, _cfg()), height=height, scrolling=False)


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
</style>
<div id="load">…</div>
<div id="tip"></div>
<script>
const CFG = __CFG__;
const C = CFG.colors;
document.body.style.background = C.bg;
document.body.style.color = C.ink;
document.getElementById("load").style.color = C.dim;
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

/* L'iframe est servie en srcdoc : son URL de base ne mène nulle part et une
   requête relative échouerait. La page hôte est de même origine, on y lit
   l'adresse des fichiers statiques. */
function staticBase() {
  try {
    const l = window.parent.location;
    return l.origin + l.pathname.replace(/\/$/, "") + "/app/static/";
  } catch (e) { return "/app/static/"; }
}

let META = null, BIN = null, Z = 0;
/* Table des séries : SERIES[cycle][mesure] est un Uint16Array de Z x pas,
   rangé zone par zone. L'ordre de lecture suit celui de l'écriture côté
   Python — cycles dans l'ordre du dictionnaire, mesures dans l'ordre de la
   liste. */
const SERIES = {}, TOPS = {};

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
    SERIES[scope] = {}; TOPS[scope] = {};
    const n = META.cycles[scope].steps;
    for (const m of META.measures) {
      const a = new Uint16Array(BIN, offset, Z * n);
      offset += Z * n * 2;
      SERIES[scope][m] = a;
      let hi = 1;
      for (let i = 0; i < a.length; i++) { if (a[i] > hi) { hi = a[i]; } }
      TOPS[scope][m] = hi;
    }
  }
  const el = document.getElementById("load");
  if (el) { el.remove(); }
}

channel.onmessage = e => {
  const m = e.data;
  if (!m || m.from === "state") { return; }
  if (m.type === "state") {
    S = m.state;
    /* Le payload peut n'être pas encore là : le premier dessin aura lieu
       dans boot(), avec le dernier état reçu. */
    if (META && typeof onState === "function") { onState(); }
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
_CONTROLS = r"""
<style>
  #wrap { position: absolute; inset: 0; display: flex; align-items: center;
          gap: 26px; padding: 0 2px; }
  .set { display: flex; gap: 7px; }
  /* Panneau de direction : tôle verte, jonc chromé en haut et en bas, texte
     blanc en capitales espacées. La même matière que la navigation. */
  .sign { position: relative; border: 0; cursor: pointer; padding: 7px 15px;
          font: inherit; font-size: 0.72rem; font-weight: 600;
          letter-spacing: 0.1em; text-transform: uppercase;
          color: rgba(244,246,244,0.62); border-radius: 2px;
          background: linear-gradient(152deg,#0a7049 0%,#00603f 46%,#004c32 100%);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.14),
                      0 1px 2px rgba(0,0,0,0.30);
          transition: color 0.18s, transform 0.18s; }
  .sign::before, .sign::after { content: ""; position: absolute; left: 0;
          right: 0; height: 2px;
          background: linear-gradient(to bottom,#8d8d97 0%,#e8e8f0 42%,
                                      #a2a2ac 62%,#6e6e78 100%); }
  .sign::before { top: 0; } .sign::after { bottom: 0; }
  .sign:hover { color: rgba(244,246,244,0.9); }
  .sign.on { color: #f4f6f4; }
  /* Phare : le halo du pas courant, posé derrière le libellé. */
  .sign.on .glow { opacity: 1; }
  .glow { position: absolute; inset: 2px 0; opacity: 0; pointer-events: none;
          background: radial-gradient(ellipse at center,
                      rgba(255,246,214,0.42) 0%, rgba(255,246,214,0) 72%);
          transition: opacity 0.25s; }
  .lab { position: relative; }
  #play { font: inherit; font-size: 0.78rem; cursor: pointer; width: 34px;
          padding: 6px 0; border-radius: 3px; }
  #now { font-size: 0.95rem; font-weight: 600; min-width: 112px;
         font-variant-numeric: tabular-nums; }
  #title { font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase; }
</style>
<div id="wrap">
  <button id="play">▶</button>
  <div class="set" id="scopes"></div>
  <div class="set" id="measures"></div>
  <div>
    <div id="title"></div>
    <div id="now"></div>
  </div>
</div>
<script>
let timer = null;

function sign(label, active, onclick) {
  const b = document.createElement("button");
  b.className = "sign" + (active ? " on" : "");
  b.innerHTML = '<span class="glow"></span><span class="lab">' + label + '</span>';
  b.onclick = onclick;
  return b;
}

function paintButtons() {
  const sc = document.getElementById("scopes");
  sc.innerHTML = "";
  for (const k of Object.keys(META.cycles)) {
    sc.appendChild(sign(META.cycles[k].label, S.scope === k, () => {
      S.scope = k; S.step = 0; S.at = Date.now(); publish();
    }));
  }
  const me = document.getElementById("measures");
  me.innerHTML = "";
  for (const m of META.measures) {
    me.appendChild(sign(MEASURE_LABEL[m], S.measure === m, () => {
      S.measure = m; publish();
    }));
  }
  document.getElementById("title").textContent = META.cycles[S.scope].title;
  document.getElementById("title").style.color = C.dim;
  document.getElementById("now").textContent =
    META.cycles[S.scope].labels[S.step];
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
  p.style.background = C.panel; p.style.color = C.ink;
  p.style.border = "1px solid " + C.rule;
  p.onclick = () => {
    S.playing = !S.playing;
    S.at = Date.now();
    publish();
  };
  publish();
}

/* Le maître de l'horloge répond aux arrivants. */
channel.addEventListener("message", e => {
  if (e.data && e.data.type === "hello" && META) { publish(); }
});
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

/* La carte garde son fond de nuit dans les deux thèmes. Une ville extrudée
   se lit par ses ombres, et une ombre n'existe pas sur fond blanc : en thème
   clair, les colonnes basses se confondaient avec le sol et la silhouette de
   New York disparaissait. Le bloc est assumé comme une fenêtre sur la ville,
   pas comme une surface de page. */
document.body.style.background = "#0b0b0f";

/* Hauteur de la plus haute colonne, en mètres. Chaque mesure est ramenée à
   son propre maximum : sans quoi la distance moyenne resterait au sol. */
const TOP_M = 9000;

/* Le bas de la rampe reste nettement au-dessus du fond : à trois heures du
   matin toutes les zones sont au plus bas, et une couleur basse trop proche
   du noir effacerait la ville au lieu de la montrer éteinte. */
const RAMP = { a: [58, 54, 67], b: [226, 172, 70] };
const EDGE = [255, 255, 255, 30];

/* Une lumière rasante donne le relief : c'est l'éclairage qui fait lire une
   hauteur, une couleur seule se lit comme une tache. L'effet est construit
   une fois deck.gl chargé, et seulement si la version en place l'expose. */
function lighting() {
  if (!deck.LightingEffect || !deck.DirectionalLight || !deck.AmbientLight) {
    return [];
  }
  return [new deck.LightingEffect({
    amb: new deck.AmbientLight({ color: [255, 248, 232], intensity: 1.05 }),
    key: new deck.DirectionalLight({ color: [255, 240, 214], intensity: 1.9,
                                     direction: [-1.2, -2.6, -1.8] }),
  })];
}

function boot() {
  deckgl = new deck.DeckGL({
    container: "map",
    /* Centré sur Manhattan et non sur le barycentre des cinq
       arrondissements : c'est là que sont 92 % des prises en charge, et
       l'inclinaison repousse de toute façon le haut de la carte au loin. */
    initialViewState: { longitude: -73.96, latitude: 40.735, zoom: 10.45,
                        pitch: CFG.pitch, bearing: -20 },
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
  frame();
}

function onState() { /* la boucle d'animation suit d'elle-même */ }

/* Les hauteurs sont interpolées entre le pas courant et le suivant : à 520 ms
   le pas, une montée instantanée saccade. L'interpolation est faite ici et
   non par les transitions de deck.gl, qui repartiraient de zéro à chaque
   changement de cycle ou de mesure. */
function frame() {
  if (deckgl && META) {
    const t = phase();
    const hi = TOPS[S.scope][S.measure];
    const rows = META.zones.map((zone, i) => {
      const a = value(i, S.step, S.scope, S.measure);
      const b = value(i, S.step + 1, S.scope, S.measure);
      return { ...zone, i: i, v: (a + (b - a) * t) / hi };
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
      material: { ambient: 0.38, diffuse: 0.78, shininess: 28,
                  specularColor: [72, 66, 54] },
      updateTriggers: { getElevation: [S.scope, S.measure, S.step, t],
                        getFillColor: [S.scope, S.measure, S.step, t] },
    })] });
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
const AHEAD = 7, BEHIND = 9;
const NOW_X = 0.42;

function size() {
  const r = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * r;
  cv.height = cv.clientHeight * r;
  ctx.setTransform(r, 0, 0, r, 0, 0);
}
window.addEventListener("resize", size);

function boot() { size(); frame(); }

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
  const span = AHEAD + BEHIND;
  const pitch = w / span;
  const bw = Math.min(pitch * 0.58, 26);
  /* `top` garde au-dessus des barres de quoi écrire la valeur du pas
     courant : sans cette réserve, le nombre sort du canvas. */
  const base = h - 34, top = 34;

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
    ctx.fillStyle = here ? C.accent : C.muted;
    ctx.fillRect(x - bw / 2, base - bh, bw, bh);

    ctx.globalAlpha = here ? 0.95 : 0.3 + fade * 0.4;
    ctx.fillStyle = here ? C.ink : C.dim;
    ctx.font = (here ? "600 " : "") + "11px system-ui, sans-serif";
    ctx.fillText(META.cycles[S.scope].labels[idx], x, base + 16);
    if (here) {
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillStyle = C.accent;
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
/* Les flux internes à un arrondissement sont écartés : à 87 % du total, ils
   écraseraient tout le reste en un seul ruban. Le bloc parle des liaisons. */
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
  const left = META.boroughs.filter(b => total["s" + b]);
  const right = META.boroughs.filter(b => total["t" + b]);
  const w = SVG.clientWidth, h = SVG.clientHeight;
  const pad = 14, gap = 11, nodeW = 11;
  const place = (list, side) => {
    const sum = list.reduce((a, b) => a + total[side + b], 0) || 1;
    const usable = h - pad * 2 - gap * (list.length - 1);
    let y = pad;
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
  LINKS.sort((a, b) => META.boroughs.indexOf(a.t) - META.boroughs.indexOf(b.t));
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

function hex(b) { return META.boroughColors[b] || C.dim; }

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
