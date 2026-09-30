# Static React build served by Nginx, which is also the single public gateway
# (docker/templates/default.conf.template). VITE_* values are public build-time config only.
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG VITE_API_BASE_URL=/api
ARG VITE_BASEMAP_STYLE_URL=https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL VITE_BASEMAP_STYLE_URL=$VITE_BASEMAP_STYLE_URL
RUN npm run build

FROM nginx:1.27-alpine
RUN apk add --no-cache openssl
# Rendered to /etc/nginx/conf.d/default.conf by the image's envsubst entrypoint step.
COPY docker/templates/default.conf.template /etc/nginx/templates/default.conf.template
# Upstream hosts: Compose service names by default; Kubernetes overrides them.
# The filter keeps nginx's own $host, $scheme, ... untouched by envsubst.
ENV SDM_UPSTREAM_API=api:8000 \
    SDM_UPSTREAM_TITILER=titiler:8000 \
    SDM_UPSTREAM_TILESERV=pg_tileserv:7800 \
    NGINX_ENVSUBST_FILTER=^SDM_UPSTREAM_
COPY --chmod=755 docker/15-sdm-resolver.sh docker/20-sdm-auth.sh /docker-entrypoint.d/
# Guard against CRLF checkouts on Windows (a CR would break the shebang).
RUN sed -i 's/\r$//' /docker-entrypoint.d/15-sdm-resolver.sh /docker-entrypoint.d/20-sdm-auth.sh
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1/healthz || exit 1
