#!/bin/sh
# Generates the gateway's Basic-auth include from env (run by the nginx image entrypoint).
# Set SDM_AUTH_USER and SDM_AUTH_PASSWORD to protect the dashboard, API and tile servers;
# leave them unset for local development.
set -eu
mkdir -p /etc/nginx/sdm
if [ -n "${SDM_AUTH_USER:-}" ] && [ -n "${SDM_AUTH_PASSWORD:-}" ]; then
  printf '%s:%s\n' "$SDM_AUTH_USER" "$(openssl passwd -apr1 "$SDM_AUTH_PASSWORD")" > /etc/nginx/sdm/htpasswd
  # Readable by the nginx worker user, not world-readable.
  chown root:nginx /etc/nginx/sdm/htpasswd
  chmod 640 /etc/nginx/sdm/htpasswd
  printf 'auth_basic "SDM dashboard";\nauth_basic_user_file /etc/nginx/sdm/htpasswd;\n' > /etc/nginx/sdm/auth.conf
  echo "sdm-auth: Basic auth enabled for user ${SDM_AUTH_USER}"
else
  : > /etc/nginx/sdm/auth.conf
  echo "sdm-auth: Basic auth disabled (SDM_AUTH_USER/SDM_AUTH_PASSWORD not set)"
fi
