"""Libellés traduits — français, anglais, espagnol.

Couvre le bandeau, la navigation et les deux onglets de contexte (`À propos`,
`Pipeline de données`). Les libellés des graphiques, dans `nyc_dashboard.py`,
sont encore en français : ils viendront une fois le texte des interprétations
figé — traduire avant qu'il soit arrêté revient à le traduire trois fois.

La langue vit dans l'URL, `?lang=`, comme la vue vit dans `?vue=` : une page
reste partageable telle qu'elle est lue, et le choix survit au rechargement
que provoque la bascule de thème. Sans paramètre, on lit l'en-tête
Accept-Language du navigateur, et à défaut le français.

Les clés sont stables et en anglais (`nav.when`), les valeurs sont le texte
affiché. Les intitulés de navigation servent aussi de valeurs à `?vue=` : ce
sont donc les clés qui voyagent dans l'URL, pas les libellés, sinon un lien
cesserait d'être valide dès qu'on change de langue.

Ne sont pas traduits, et n'ont pas à l'être : les noms de tables, de colonnes,
de fichiers et de produits (`daily_summary`, `read_parquet()`, Snowflake,
dbt Core). Les traduire ferait mentir le schéma, qui doit nommer ce qui existe
réellement dans le dépôt.
"""

import streamlit as st

DEFAULT = "fr"
LANGS = ("fr", "en", "es")

# Séparateur de milliers. Insécable en français et en espagnol — l'Académie
# comme la RAE proscrivent le point, qui se lit comme une virgule décimale
# d'un pays à l'autre. L'anglais garde la virgule.
GROUP = {"fr": " ", "en": ",", "es": " "}

STRINGS = {
    "fr": {
        "band.title": "Analyse des courses de taxi",
        "nav.how_much": "Combien",
        "nav.when": "Quand",
        "nav.where": "Où",
        "nav.how": "Comment",
        "nav.about": "À propos",
        "nav.pipeline": "Pipeline de données",

        "about.period": "janvier 2024 – septembre 2025",
        "about.k.project": "Projet",
        "about.v.project": "Projet étudiant — pipeline de données bout-en-bout",
        "about.k.goal": "Objectif",
        "about.v.goal": "Éprouver Snowflake et dbt Core sur un volume réaliste",
        "about.k.data": "Données",
        "about.v.data": "{rows} courses de taxis jaunes new-yorkais",
        "about.v.data_sub": "NYC TLC Open Data · {period}",
        "about.k.arch": "Architecture",
        "about.v.arch_sub": "modèles dbt versionnés, testés à chaque build",
        "about.k.output": "Restitution",
        "about.next.pipeline": "le détail de la chaîne, étape par étape",
        "about.next.dashboard": "l'analyse menée sur les tables finales",
        "about.repo": "Code source sur GitHub",

        "pipe.h.flow": "Circulation de la donnée",
        "pipe.intro": "Deux chemins partent de la même source. Le premier est "
                      "l'architecture conçue pour le projet, aujourd'hui éteinte ; "
                      "le second alimente réellement les graphiques de l'onglet "
                      "Dashboard. Les étages logiques sont les mêmes — seul le "
                      "moteur de calcul change. Le point lumineux indique le sens "
                      "de circulation ; chaque vignette montre un extrait réel de "
                      "ce que contient l'étage.",
        "pipe.rail.past": "Architecture d'origine — Snowflake + dbt Core",
        "pipe.rail.past.badge": "Warehouse arrêté",
        "pipe.rail.live": "Chemin actif — copie Parquet sur VPS + DuckDB",
        "pipe.rail.live.badge": "En service",
        "pipe.why.title": "Pourquoi deux rails ?",
        "pipe.why.body": "Le warehouse Snowflake était facturé à l'usage et n'est "
                         "plus financé depuis la fin de la formation. Les tables "
                         "<code>FINAL</code> ont donc été exportées en Parquet et "
                         "déposées sur un VPS loué. Les étages logiques n'ont pas "
                         "bougé — ingestion, nettoyage, agrégation, restitution — "
                         "seul le moteur qui les exécute a changé. Les chiffres du "
                         "Dashboard sont les mêmes qu'en sortie de warehouse.",

        "pipe.h.schema": "Schéma FINAL — tables analytiques",
        "pipe.kind.agg": "Agrégat",
        "pipe.kind.staging": "Staging",
        "pipe.tbl.clean": "trajets filtrés (0 &lt; distance ≤ 100 mi, montants &gt; 0, "
                          "microsecondes corrigées)",

        "pipe.stage.vps_copy": "Copie VPS",
        "pipe.stage.aggregates": "Agrégats",
        "pipe.sub.marts": "Marts agrégés",
        "pipe.sub.same_parquet": "Mêmes Parquet",
        "pipe.sub.docker_volume": "Volume Docker",
        "pipe.sub.disk_cache": "Cache disque",

        "pipe.p.raw.rows": "lignes",
        "pipe.p.raw.load": "charge",
        "pipe.p.raw.note": "Copié tel quel, horodatages aberrants compris "
                           "(2002 → 2026 dans la source TLC).",
        "pipe.p.stg.kept": "retenues",
        "pipe.p.stg.note": "&minus;{dropped} lignes écartées, dont {outliers} courses de "
                           "plus de 100 mi (jusqu'à {max_mi} mi : erreurs GPS qui "
                           "doublaient la distance moyenne). + tests dbt_utils.",
        "pipe.p.copy.files": "fichiers",
        "pipe.p.copy.mount": "montage",
        "pipe.p.copy.note": "Téléchargés une fois par entrypoint.sh.",
        "pipe.p.duck.title": "Scan en mémoire",
        "pipe.p.duck.note": "Mêmes requêtes, pas de warehouse.",
        "pipe.p.cache.note": "Persisté : un redémarrage ne rescanne pas.",
        "pipe.p.here.badge": "Vous êtes ici",
        "pipe.p.here.note": "Streamlit + Plotly, requêtes mises en cache.",
        "pipe.p.offline.title": "Dashboard sur Snowflake",
        "pipe.p.offline.note": "Le dashboard lisait les tables FINAL directement "
                               "dans le warehouse.",

        "pipe.h.infra": "Les deux infrastructures",
        "pipe.infra.past.state": "Architecture d'origine",
        "pipe.infra.live.state": "En service",
        "pipe.infra.past.1": "Warehouse <code>NYC_TAXI_WH</code>, DB <code>NYC_TAXI_DB</code>",
        "pipe.infra.past.2": "Schémas <code>RAW</code> / <code>STAGING</code> / <code>FINAL</code>",
        "pipe.infra.past.3": "Chargement par <code>COPY INTO</code> depuis les Parquet TLC",
        "pipe.infra.past.4": "Modèles dbt + tests automatiques (dbt_utils)",
        "pipe.infra.live.1": "Tables <code>FINAL</code> exportées en Parquet",
        "pipe.infra.live.2": "Scan direct des fichiers, sans base distante",
        "pipe.infra.live.3": "Conteneur Docker derrière Caddy (reverse proxy)",
        "pipe.infra.live.4": "Volume bind-mount, téléchargement au 1ᵉʳ démarrage",
    },
    "en": {
        "band.title": "Taxi ride analysis",
        "nav.how_much": "How much",
        "nav.when": "When",
        "nav.where": "Where",
        "nav.how": "How",
        "nav.about": "About",
        "nav.pipeline": "Data pipeline",

        "about.period": "January 2024 – September 2025",
        "about.k.project": "Project",
        "about.v.project": "Student project — end-to-end data pipeline",
        "about.k.goal": "Goal",
        "about.v.goal": "Put Snowflake and dbt Core through a realistic volume",
        "about.k.data": "Data",
        "about.v.data": "{rows} New York yellow taxi rides",
        "about.v.data_sub": "NYC TLC Open Data · {period}",
        "about.k.arch": "Architecture",
        "about.v.arch_sub": "versioned dbt models, tested on every build",
        "about.k.output": "Front end",
        "about.next.pipeline": "the chain in detail, step by step",
        "about.next.dashboard": "the analysis run on the final tables",
        "about.repo": "Source code on GitHub",

        "pipe.h.flow": "How the data moves",
        "pipe.intro": "Two paths leave the same source. The first is the "
                      "architecture designed for the project, now shut down; the "
                      "second is what actually feeds the charts in the Dashboard "
                      "tab. The logical stages are identical — only the compute "
                      "engine changes. The travelling dot shows the direction of "
                      "flow; each card shows a real sample of what the stage holds.",
        "pipe.rail.past": "Original architecture — Snowflake + dbt Core",
        "pipe.rail.past.badge": "Warehouse shut down",
        "pipe.rail.live": "Live path — Parquet copy on a VPS + DuckDB",
        "pipe.rail.live.badge": "Running",
        "pipe.why.title": "Why two rails?",
        "pipe.why.body": "The Snowflake warehouse was billed per use and has not "
                         "been funded since the course ended. The <code>FINAL</code> "
                         "tables were therefore exported to Parquet and placed on a "
                         "rented VPS. The logical stages did not move — ingestion, "
                         "cleaning, aggregation, presentation — only the engine "
                         "running them changed. The Dashboard figures are the ones "
                         "the warehouse produced.",

        "pipe.h.schema": "FINAL schema — analytics tables",
        "pipe.kind.agg": "Aggregate",
        "pipe.kind.staging": "Staging",
        "pipe.tbl.clean": "filtered rides (0 &lt; distance ≤ 100 mi, amounts &gt; 0, "
                          "microseconds fixed)",

        "pipe.stage.vps_copy": "VPS copy",
        "pipe.stage.aggregates": "Aggregates",
        "pipe.sub.marts": "Aggregated marts",
        "pipe.sub.same_parquet": "Same Parquet",
        "pipe.sub.docker_volume": "Docker volume",
        "pipe.sub.disk_cache": "Disk cache",

        "pipe.p.raw.rows": "rows",
        "pipe.p.raw.load": "load",
        "pipe.p.raw.note": "Copied as is, stray timestamps included "
                           "(2002 → 2026 in the TLC source).",
        "pipe.p.stg.kept": "kept",
        "pipe.p.stg.note": "&minus;{dropped} rows dropped, among them {outliers} rides "
                           "over 100 mi (up to {max_mi} mi: GPS errors that doubled "
                           "the average distance). + dbt_utils tests.",
        "pipe.p.copy.files": "files",
        "pipe.p.copy.mount": "mount",
        "pipe.p.copy.note": "Downloaded once by entrypoint.sh.",
        "pipe.p.duck.title": "In-memory scan",
        "pipe.p.duck.note": "Same queries, no warehouse.",
        "pipe.p.cache.note": "Persisted: a restart does not rescan.",
        "pipe.p.here.badge": "You are here",
        "pipe.p.here.note": "Streamlit + Plotly, queries cached.",
        "pipe.p.offline.title": "Dashboard on Snowflake",
        "pipe.p.offline.note": "The dashboard read the FINAL tables straight from "
                               "the warehouse.",

        "pipe.h.infra": "The two infrastructures",
        "pipe.infra.past.state": "Original architecture",
        "pipe.infra.live.state": "Running",
        "pipe.infra.past.1": "<code>NYC_TAXI_WH</code> warehouse, <code>NYC_TAXI_DB</code> database",
        "pipe.infra.past.2": "<code>RAW</code> / <code>STAGING</code> / <code>FINAL</code> schemas",
        "pipe.infra.past.3": "Loaded with <code>COPY INTO</code> from the TLC Parquet files",
        "pipe.infra.past.4": "dbt models + automated tests (dbt_utils)",
        "pipe.infra.live.1": "<code>FINAL</code> tables exported to Parquet",
        "pipe.infra.live.2": "Files scanned directly, no remote database",
        "pipe.infra.live.3": "Docker container behind Caddy (reverse proxy)",
        "pipe.infra.live.4": "Bind-mounted volume, downloaded on first start",
    },
    "es": {
        "band.title": "Análisis de carreras de taxi",
        "nav.how_much": "Cuánto",
        "nav.when": "Cuándo",
        "nav.where": "Dónde",
        "nav.how": "Cómo",
        "nav.about": "Acerca de",
        "nav.pipeline": "Flujo de datos",

        "about.period": "enero de 2024 – septiembre de 2025",
        "about.k.project": "Proyecto",
        "about.v.project": "Proyecto estudiantil — flujo de datos de extremo a extremo",
        "about.k.goal": "Objetivo",
        "about.v.goal": "Poner a prueba Snowflake y dbt Core con un volumen realista",
        "about.k.data": "Datos",
        "about.v.data": "{rows} carreras de taxis amarillos de Nueva York",
        "about.v.data_sub": "NYC TLC Open Data · {period}",
        "about.k.arch": "Arquitectura",
        "about.v.arch_sub": "modelos dbt versionados, probados en cada build",
        "about.k.output": "Presentación",
        "about.next.pipeline": "la cadena en detalle, paso a paso",
        "about.next.dashboard": "el análisis realizado sobre las tablas finales",
        "about.repo": "Código fuente en GitHub",

        "pipe.h.flow": "Circulación de los datos",
        "pipe.intro": "Dos caminos parten de la misma fuente. El primero es la "
                      "arquitectura diseñada para el proyecto, hoy apagada; el "
                      "segundo alimenta realmente los gráficos de la pestaña "
                      "Dashboard. Las etapas lógicas son las mismas — solo cambia "
                      "el motor de cálculo. El punto luminoso indica el sentido de "
                      "la circulación; cada ficha muestra un extracto real de lo "
                      "que contiene la etapa.",
        "pipe.rail.past": "Arquitectura original — Snowflake + dbt Core",
        "pipe.rail.past.badge": "Warehouse apagado",
        "pipe.rail.live": "Camino activo — copia Parquet en un VPS + DuckDB",
        "pipe.rail.live.badge": "En servicio",
        "pipe.why.title": "¿Por qué dos vías?",
        "pipe.why.body": "El warehouse de Snowflake se facturaba por uso y ya no "
                         "está financiado desde que terminó la formación. Las tablas "
                         "<code>FINAL</code> se exportaron por tanto a Parquet y se "
                         "colocaron en un VPS alquilado. Las etapas lógicas no se "
                         "movieron — ingesta, limpieza, agregación, presentación — "
                         "solo cambió el motor que las ejecuta. Las cifras del "
                         "Dashboard son las que salían del warehouse.",

        "pipe.h.schema": "Esquema FINAL — tablas analíticas",
        "pipe.kind.agg": "Agregado",
        "pipe.kind.staging": "Staging",
        "pipe.tbl.clean": "trayectos filtrados (0 &lt; distancia ≤ 100 mi, importes "
                          "&gt; 0, microsegundos corregidos)",

        "pipe.stage.vps_copy": "Copia VPS",
        "pipe.stage.aggregates": "Agregados",
        "pipe.sub.marts": "Marts agregados",
        "pipe.sub.same_parquet": "Mismos Parquet",
        "pipe.sub.docker_volume": "Volumen Docker",
        "pipe.sub.disk_cache": "Caché en disco",

        "pipe.p.raw.rows": "filas",
        "pipe.p.raw.load": "carga",
        "pipe.p.raw.note": "Copiado tal cual, con marcas de tiempo aberrantes "
                           "incluidas (2002 → 2026 en la fuente TLC).",
        "pipe.p.stg.kept": "retenidas",
        "pipe.p.stg.note": "&minus;{dropped} filas descartadas, entre ellas {outliers} "
                           "carreras de más de 100 mi (hasta {max_mi} mi: errores de "
                           "GPS que duplicaban la distancia media). + pruebas dbt_utils.",
        "pipe.p.copy.files": "archivos",
        "pipe.p.copy.mount": "montaje",
        "pipe.p.copy.note": "Descargados una vez por entrypoint.sh.",
        "pipe.p.duck.title": "Escaneo en memoria",
        "pipe.p.duck.note": "Mismas consultas, sin warehouse.",
        "pipe.p.cache.note": "Persistido: un reinicio no vuelve a escanear.",
        "pipe.p.here.badge": "Estás aquí",
        "pipe.p.here.note": "Streamlit + Plotly, consultas en caché.",
        "pipe.p.offline.title": "Dashboard en Snowflake",
        "pipe.p.offline.note": "El dashboard leía las tablas FINAL directamente en "
                               "el warehouse.",

        "pipe.h.infra": "Las dos infraestructuras",
        "pipe.infra.past.state": "Arquitectura original",
        "pipe.infra.live.state": "En servicio",
        "pipe.infra.past.1": "Warehouse <code>NYC_TAXI_WH</code>, BD <code>NYC_TAXI_DB</code>",
        "pipe.infra.past.2": "Esquemas <code>RAW</code> / <code>STAGING</code> / <code>FINAL</code>",
        "pipe.infra.past.3": "Carga con <code>COPY INTO</code> desde los Parquet de TLC",
        "pipe.infra.past.4": "Modelos dbt + pruebas automáticas (dbt_utils)",
        "pipe.infra.live.1": "Tablas <code>FINAL</code> exportadas a Parquet",
        "pipe.infra.live.2": "Escaneo directo de los archivos, sin base remota",
        "pipe.infra.live.3": "Contenedor Docker detrás de Caddy (reverse proxy)",
        "pipe.infra.live.4": "Volumen bind-mount, descarga en el primer arranque",
    },
}


def _from_browser() -> str:
    """Première langue connue de l'en-tête Accept-Language, sinon DEFAULT.

    L'en-tête est une liste pondérée — `fr-CA,fr;q=0.9,en;q=0.8`. On ne lit
    que le code de langue de chaque entrée, dans l'ordre : les pondérations y
    sont déjà rangées par ordre de préférence, et un tri sur `q` ne changerait
    le résultat que pour des en-têtes mal formés.
    """
    try:
        header = st.context.headers.get("Accept-Language", "")
    except Exception:
        return DEFAULT
    for part in header.split(","):
        code = part.split(";")[0].strip().lower()[:2]
        if code in LANGS:
            return code
    return DEFAULT


def current() -> str:
    """Langue en vigueur : l'URL si elle en nomme une, sinon le navigateur."""
    asked = st.query_params.get("lang")
    return asked if asked in LANGS else _from_browser()


def t(key: str, **kwargs) -> str:
    """Texte de `key` dans la langue courante, interpolé si besoin.

    Le français sert de recours : une clé ajoutée mais pas encore traduite
    s'affiche en français plutôt que de faire tomber la page, et se repère à
    l'œil sur la version traduite.
    """
    lang = current()
    text = STRINGS[lang].get(key) or STRINGS[DEFAULT].get(key, key)
    return text.format(**kwargs) if kwargs else text


def n(value: int) -> str:
    """Entier groupé par milliers selon la langue, séparateur insécable.

    Insécable et non simple espace : un nombre coupé en fin de ligne se lit
    comme deux nombres.
    """
    return f"{value:,}".replace(",", GROUP.get(current(), GROUP[DEFAULT]))
