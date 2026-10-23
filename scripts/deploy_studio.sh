#!/bin/sh
# Deploy only the RAG container, retaining the previous container for rollback.
set -eu
export AWS_DEFAULT_REGION=us-east-1
REGISTRY=597303283286.dkr.ecr.us-east-1.amazonaws.com
IMAGE=${IMAGE_URI:-$REGISTRY/rag-ai-demo:latest}
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker pull "$IMAGE"
install -d -m 700 /opt/rag-studio
install -d -o 10001 -g 10001 /opt/rag-studio/storage
umask 077
aws ssm get-parameters --names /rag-studio/openai-api-key --with-decryption --query 'Parameters[0].Value' --output text > /opt/rag-studio/key.tmp
python3 - <<'PY'
from pathlib import Path
p = Path('/opt/rag-studio/key.tmp')
key = p.read_text().strip()
if not key or key == 'None' or '\n' in key:
    raise SystemExit('Missing or invalid deployment credential')
Path('/opt/rag-studio/runtime.env').write_text('OPENAI_API_KEY=' + key + '\n')
p.unlink()
PY
OLD=rag-ai-app-backup-$(date +%s)
docker stop rag-ai-app
docker rename rag-ai-app "$OLD"
if docker run -d --name rag-ai-app --restart unless-stopped --env-file /opt/rag-studio/runtime.env -v /opt/rag-studio/storage:/app/storage -p 3002:8080 "$IMAGE"; then
    for i in $(seq 1 60); do
        if curl -fsS http://127.0.0.1:3002/api/generic/status >/dev/null; then
            echo "RAG Studio API healthy; rollback container: $OLD"
            exit 0
        fi
        sleep 3
    done
fi
docker logs --tail 25 rag-ai-app
docker stop rag-ai-app || true
docker rename rag-ai-app "rag-ai-app-failed-$(date +%s)"
docker rename "$OLD" rag-ai-app
docker start rag-ai-app
exit 1
