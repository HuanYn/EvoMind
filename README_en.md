# EvoMind

EvoMind is a personal, reproducible small Chinese LLM and multimodal-extension project maintained by YH. The primary documentation is available in [README.md](README.md).

The repository contains code, configs, tests, and documentation for:

- decoder-only pretraining and assistant-only SFT;
- DPO and controlled GRPO/CISPO post-training comparisons;
- checkpoint/data/evaluation contracts and diagnostics;
- a local Streamlit text UI; and
- an upcoming Dense VLM path (single image, multi-image, video baseline) on the `evomind-v` branch.

It does not ship training datasets, reward-model weights, or EvoMind checkpoints. Current text-model results are intentionally reported with limitations: small post-training deltas do not demonstrate a general capability gain, and the project does not claim reliable knowledge QA, validated long-context ability, or completed VLM/video capability.

## Minimal setup

```bash
git clone https://github.com/HuanYn/evomind.git
cd evomind
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
# install a CUDA-compatible PyTorch build first
python -m pip install -r requirements-minimal.txt

python -B scripts/evomind_reproduce.py smoke
```

For the full training/data/evaluation contract, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). On Windows, start the local UI with:

```powershell
.venv\Scripts\python.exe -m pip install streamlit==1.50.0
.venv\Scripts\python.exe -m streamlit run scripts\evomind_webui.py
```

## Acknowledgement

EvoMind is a personal reproduction and extension by YH, based on the open-source [MiniMind](https://github.com/jingyaogong/minimind) model code and training recipes. The experiments, engineering adaptations, and result records described here belong to this project. See [LICENSE](LICENSE) and [NOTICE](NOTICE.md) for attribution and terms.
