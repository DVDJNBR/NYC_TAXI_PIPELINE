#!/bin/sh
set -e

if [ -z "$(ls -A /app/data/yellow_taxi/*.parquet 2>/dev/null)" ]; then
    echo "Aucune donnée trouvée dans /app/data/yellow_taxi — téléchargement initial (peut prendre plusieurs minutes)..."
    python scripts/B_load_local_parquet.py
fi

if [ -z "$(ls -A /app/data/aggregates/*.parquet 2>/dev/null)" ]; then
    echo "Pré-calcul des agrégats (une seule fois, ~1min)..."
    python -c "from streamlit_dashboard_local import load_data; load_data()"
fi

# Les agrégats du cockpit sont d'une autre fabrication que ceux du
# dashboard : ils croisent la zone et le pas du cycle, et c'est J qui les
# écrit, après G. Sans eux le cockpit n'affiche que son message d'absence,
# ce qui en production revient à une page vide.
if [ ! -f /app/data/aggregates/zone_cycle.parquet ] \
   || [ ! -f /app/data/aggregates/od_cycle.parquet ]; then
    echo "Pré-calcul des cycles du cockpit (une seule fois, ~2min)..."
    python scripts/G_build_aggregates.py
    python scripts/J_build_cycles.py
fi

exec streamlit run streamlit_dashboard_local.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
