#!/bin/sh
# Writes the gateway's DNS resolver include (run by the nginx image entrypoint).
# Upstreams are proxied through variables so nginx re-resolves them at request time: a
# recreated api / titiler / pg_tileserv container gets a new IP, and a name resolved once at
# startup would keep pointing at the old one (502 until the gateway restarted).
# SDM_DNS_RESOLVER overrides the nameserver; by default the container's own is used
# (Docker's embedded DNS 127.0.0.11 under Compose, kube-dns under Kubernetes).
set -eu
mkdir -p /etc/nginx/sdm
ns="${SDM_DNS_RESOLVER:-$(awk '$1 == "nameserver" { print $2; exit }' /etc/resolv.conf)}"
if [ -z "$ns" ]; then
  echo "sdm-resolver: no nameserver found; set SDM_DNS_RESOLVER" >&2
  exit 1
fi
case "$ns" in
  *:*) ns="[$ns]" ;;  # IPv6 literal
esac
printf 'resolver %s valid=10s;\nresolver_timeout 5s;\n' "$ns" > /etc/nginx/sdm/resolver.conf
echo "sdm-resolver: upstreams resolved per request via ${ns}"
