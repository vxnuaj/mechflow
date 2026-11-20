#!/bin/bash
# Run ON the pod once. Expects ~/mf (the mechflow repo at the pinned commit) and ~/hcfrl_min (hcfrl/elf_path.py + hcfrl/elf).
set -euo pipefail
cd ~; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
uv venv -q --python 3.12 ~/venv; source ~/venv/bin/activate
uv pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -q -r ~/mf/pod/requirements_e4.txt
uv pip freeze > ~/mf/pod/pod_freeze.txt
export HF_HOME=$HOME/hf
python - <<'PY'
from huggingface_hub import snapshot_download as s, hf_hub_download as h
for repo, rev in [("embedded-language-flows/ELF-B-owt-torch", "146f84133c1389bfd4ef47f14ec7a955da22faa7"),
                  ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
                  ("Continuous-Rivals-Discrete/langflow-owt", "a08f933dd337d52762fec5ef7d60c131896cc341"),
                  ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"), ("t5-small", "df1b051c49625cf57a3d0d8d3863ed4d13564fe4")]:
    print(repo, s(repo, revision=rev, allow_patterns=None if "Mistral" not in repo else ["*.json", "*.safetensors", "tokenizer*"]))
import os; d = os.path.expanduser("~/data/owt_t5"); os.makedirs(d, exist_ok=True)
for f in ["data-00000-of-00075.arrow", "data-00001-of-00075.arrow", "dataset_info.json", "state.json"]:
    h("embedded-language-flows/openwebtext-t5", f, repo_type="dataset", local_dir=d)
import json; st = json.load(open(f"{d}/state.json")); json.dump(st, open(f"{d}/state_all75.json", "w"))
st["_data_files"] = [x for x in st["_data_files"] if x["filename"] in ("data-00000-of-00075.arrow", "data-00001-of-00075.arrow")]
json.dump(st, open(f"{d}/state.json", "w")); print("OWT ok")
PY
echo SETUP_DONE
