FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY streamlit_dashboard_local.py .
COPY nyc_theme.py nyc_content.py nyc_dashboard.py .
COPY scripts/B_load_local_parquet.py scripts/B_load_local_parquet.py
COPY .streamlit/ .streamlit/
COPY assets/ assets/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
