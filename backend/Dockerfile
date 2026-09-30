# Backend image: FastAPI API, Celery worker/beat and Prefect flows share this image.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# WeasyPrint needs Pango/HarfBuzz; fonts for the bulletin. rasterio wheels bundle GDAL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0 \
        libffi8 shared-mime-info fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
# Dependencies in their own layer (rebuilt only when pyproject.toml changes): install against an
# empty package stub, then add the real source without touching dependencies.
RUN mkdir src && touch src/__init__.py \
    && pip install ".[dev]" && pip uninstall -y sdm-dashboard-backend && rm -rf src build
COPY src ./src
RUN pip install --no-deps ".[dev]" && rm -rf build

COPY db ./db
COPY tests ./tests

RUN useradd --create-home --uid 1000 sdm && mkdir -p /data /artifacts && chown -R sdm /app /data /artifacts
USER sdm

ENV SDM_DATA_ROOT=/data \
    SDM_ARTIFACT_ROOT=/artifacts

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
