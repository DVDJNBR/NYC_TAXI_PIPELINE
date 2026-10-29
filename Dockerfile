FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY streamlit_dashboard_local.py .
# Les huit modules de l'application. La liste était restée à trois : le
# cockpit, les composants animés, le référentiel des zones, les libellés et
# les vues ont été ajoutés depuis, et l'image se construisait sans eux —
# donc un ModuleNotFoundError au premier import, en production seulement.
COPY nyc_cockpit.py nyc_content.py nyc_dashboard.py nyc_lang.py \
     nyc_live.py nyc_theme.py nyc_views.py nyc_zones.py ./
# Géométrie de référence : les contours des 263 zones et la maquette de la
# ville. Versionnées avec le dépôt (voir l'exception dans .gitignore) parce
# qu'elles sont stables et que le cockpit en a besoin au démarrage.
COPY data/reference/ data/reference/
# Les trois scripts que l'entrypoint peut avoir à lancer au premier
# démarrage : le téléchargement, les agrégats journaliers, les cycles.
COPY scripts/B_load_local_parquet.py scripts/G_build_aggregates.py \
     scripts/J_build_cycles.py scripts/
COPY .streamlit/ .streamlit/
COPY assets/ assets/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
