#!/usr/bin/env bash
# Run from the user's network-enabled terminal; uses the clone URL in GitCode's page.
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export GIT_TERMINAL_PROMPT=0
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task_repo="$task_root/data/bones-seed-gitcode"
mkdir -p "$task_root/data" "$task_root/logs"
exec > >(tee -a "$task_root/logs/bones-gitcode-download.log") 2>&1
git lfs version
if [[ ! -d "$task_repo/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -c http.proxy= clone --depth 1 --single-branch --branch main \
        https://atomgit.com/hf_mirrors/bones-studio/seed.git "$task_repo"
fi
cd "$task_repo"
task_remote="$(git remote get-url origin)"
if [[ "$task_remote" != 'https://atomgit.com/hf_mirrors/bones-studio/seed.git' ]]; then
    echo 'Existing destination has a different remote; refusing to alter it.'
    exit 1
fi
# Stop if the repository explicitly directs LFS back to Hugging Face.
if [[ -f .lfsconfig ]] && rg -qi 'huggingface\.co|hf\.co|hf-mirror' .lfsconfig; then
    echo 'Repository LFS configuration points to Hugging Face; stopped.'
    exit 1
fi
git -c http.proxy= fetch --depth 1 origin 3347a32c4542a0a1165d6c4667ef631df2bc4c9d
git checkout --detach FETCH_HEAD
git rev-parse HEAD
git lfs install --local
echo 'Fetching G1 + SOMA Uniform + metadata/shape assets (approximately 69 GB of motion archives).'
git -c http.proxy= -c lfs.concurrenttransfers=2 -c lfs.transfer.maxretries=5 \
    lfs pull --include='g1.tar.gz,soma_uniform.tar.gz,metadata/**,soma_shapes/**,LICENSE.md' --exclude=''
python3 - <<'PY'
from pathlib import Path
import json
for name in ['g1.tar.gz', 'soma_uniform.tar.gz']:
    p = Path(name)
    if not p.is_file():
        raise SystemExit(f'Missing archive: {name}')
    with p.open('rb') as stream:
        magic = stream.read(2)
    if magic != b'\x1f\x8b':
        raise SystemExit(f'{name} is not a gzip archive (possibly an LFS pointer or error page)')
    print(json.dumps({'file': str(p.resolve()), 'bytes': p.stat().st_size}))
PY
echo "Download finished: $task_repo"
