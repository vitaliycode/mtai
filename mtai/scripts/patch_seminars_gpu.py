#!/usr/bin/env python3
"""Patch seminar 4/5 notebooks for GPU runs (conda cu124, no ~/.local torch)."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPS = REPO / ".seminar_deps"

GPU_SETUP = f'''import os
import site
import sys

# Use conda PyTorch (cu124) — ignore ~/.local torch (cu130 breaks this GPU driver)
os.environ["PYTHONNOUSERSITE"] = "1"
site.ENABLE_USER_SITE = False
sys.path = [p for p in sys.path if "/.local/" not in p.replace("\\\\", "/")]
_deps = {repr(str(DEPS))}
if os.path.isdir(_deps) and _deps not in sys.path:
    sys.path.append(_deps)  # after conda env so bundled torch is not preferred
'''

PIP_CELL = f'''import os
import subprocess
import sys

_deps = {repr(str(DEPS))}
os.makedirs(_deps, exist_ok=True)
env = os.environ.copy()
env["PYTHONNOUSERSITE"] = "1"
for pkg in ("torchprofile", "torchao", "nncf"):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--target", _deps, pkg, "-q"],
        env=env,
        check=False,
    )
print("Dependencies installed to", _deps)
'''


BASELINE_TRAIN_CELL = '''# Dense baseline: load cached / MIT checkpoint, or train on GPU
from pathlib import Path

_assets = Path('assets')
_assets.mkdir(exist_ok=True)
_cached = _assets / 'vgg_cifar_baseline.pth'

if _cached.is_file():
    _ck = torch.load(_cached, map_location='cpu')
    model.load_state_dict(_ck['state_dict'])
    checkpoint = {'state_dict': copy.deepcopy(model.state_dict())}
    recover_model = lambda: model.load_state_dict(checkpoint['state_dict'])
    print(f'=> loaded cached baseline ({_cached}), acc={evaluate(model, dataloader["test"]):.2f}%')
else:
    _dense_acc = evaluate(model, dataloader['test'])
    if _dense_acc < 80.0:
        print(f'=> dense accuracy {_dense_acc:.2f}% — training baseline on {DEVICE}...')
        _opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        _sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=200)
        _crit = nn.CrossEntropyLoss()
        _best_acc, _best_state = _dense_acc, None
        for _epoch in range(200):
            train(model, dataloader['train'], _crit, _opt, _sched)
            _acc = evaluate(model, dataloader['test'])
            if _acc > _best_acc:
                _best_acc, _best_state = _acc, copy.deepcopy(model.state_dict())
            if (_epoch + 1) % 20 == 0 or _acc >= 92.5:
                print(f'    epoch {_epoch + 1}: acc={_acc:.2f}% best={_best_acc:.2f}%')
            if _best_acc >= 92.5:
                break
        if _best_state is not None:
            model.load_state_dict(_best_state)
        checkpoint = {'state_dict': copy.deepcopy(model.state_dict())}
        torch.save(checkpoint, _cached)
        print(f'=> saved baseline to {_cached}')
    else:
        checkpoint = {'state_dict': copy.deepcopy(model.state_dict())}
    recover_model = lambda: model.load_state_dict(checkpoint['state_dict'])
    print(f'=> dense baseline accuracy: {evaluate(model, dataloader["test"]):.2f}%')
'''


def patch_pruning(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text())
    inserted_baseline = False
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and src.startswith("# Prunning"):
            cell["source"] = [
                "# Pruning\n",
                "\n",
                "Based on **MIT 6.5940 EfficientML.ai Fall 2023: Lab 1 Pruning**\n",
                "\n",
                "> **GPU:** run Jupyter with `PYTHONNOUSERSITE=1` or use `scripts/run_seminar_gpu.sh`.\n",
            ]
        if "print('Installing torchprofile')" in src:
            cell["source"] = [PIP_CELL]
        if "from torchprofile import profile_macs" in src and "PYTHONNOUSERSITE" not in src:
            cell["source"] = [GPU_SETUP + "\n" + src]
        if "next_conv.weight.copy_(0)" in src:
            cell["source"] = [
                s.replace(
                    "next_conv.weight.copy_(0)",
                    "next_conv.weight.copy_(torch.index_select(next_conv.weight.detach(), 1, sort_idx))",
                )
                for s in cell["source"]
            ]
        # Q1 answers
        if "**Your Answer:** Pruning removes low-importance" in src:
            cell["source"] = [
                "**Your Answer:** Weights are roughly bell-shaped and centered near zero in most layers; "
                "later layers often have larger magnitude spread. Many small-magnitude weights are good "
                "candidates for pruning.\n"
            ]
        if "**Your Answer:** Fine-grained pruning produces sparse tensors" in src and "Question 1.2" in "".join(
            nb["cells"][nb["cells"].index(cell) - 1].get("source", [])
        ):
            cell["source"] = [
                "**Your Answer:** Small magnitudes near zero can be removed with limited impact; "
                "large-magnitude weights carry more signal, so magnitude pruning is effective.\n"
            ]
        if "target_sparsity = 0.6 # please modify" in src:
            cell["source"] = [
                s.replace(
                    "target_sparsity = 0.6 # please modify the value of target_sparsity",
                    "target_sparsity = 0.6  # 25 elements -> 10 nonzeros",
                )
                for s in cell["source"]
            ]
        if "urlretrieve(url, cached_file)" in src and "timeout" not in src:
            cell["source"] = [
                s.replace(
                    "urlretrieve(url, cached_file)",
                    "urlretrieve(url, cached_file, timeout=120)",
                )
                for s in cell["source"]
            ]

        if (
            not inserted_baseline
            and "## Let's First Evaluate the Accuracy" in src
            and cell["cell_type"] == "markdown"
        ):
            nb["cells"].insert(
                i,
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [BASELINE_TRAIN_CELL],
                    "outputs": [],
                    "execution_count": None,
                },
            )
            inserted_baseline = True

    _clear_outputs(nb)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def _clear_outputs(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def patch_quantization(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text())
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and src.strip() == "import torch":
            cell["source"] = [GPU_SETUP + "\nimport torch\nprint('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"]
        if "Int4DynamicActivationInt4WeightConfig" in src:
            cell["source"] = [
                "try:\n",
                "    from torchao.quantization import quantize_\n",
                "    try:\n",
                "        from torchao.quantization import Int4WeightOnlyConfig as _Int4Cfg\n",
                "    except ImportError:\n",
                "        from torchao.quantization import Float8DynamicActivationInt4WeightConfig as _Int4Cfg\n",
                "    quantize_(model, _Int4Cfg())\n",
                "    print('torchao int4/weight-only quantization applied:', _Int4Cfg.__name__)\n",
                "except Exception as e:\n",
                "    print(f'torchao quantization skipped: {e}')\n",
            ]
        if "benchmark_model(model_f32" in src:
            cell["source"] = [
                "import time\n",
                "\n",
                "def _bench_ms(fn, example_inputs, n_warmup=10, n_runs=50):\n",
                "    fn.eval()\n",
                "    with torch.inference_mode():\n",
                "        for _ in range(n_warmup):\n",
                "            fn(*example_inputs)\n",
                "        if torch.cuda.is_available():\n",
                "            torch.cuda.synchronize()\n",
                "        t0 = time.perf_counter()\n",
                "        for _ in range(n_runs):\n",
                "            fn(*example_inputs)\n",
                "        if torch.cuda.is_available():\n",
                "            torch.cuda.synchronize()\n",
                "        return (time.perf_counter() - t0) / n_runs * 1000\n",
                "\n",
                "try:\n",
                "    example_inputs = (torch.randn(1, 1024, device=next(model.parameters()).device),)\n",
                "    f32_time = _bench_ms(model_f32, example_inputs)\n",
                "    int4_time = _bench_ms(model, example_inputs)\n",
                "    print('f32 mean time: %.3f ms' % f32_time)\n",
                "    print('quantized mean time: %.3f ms' % int4_time)\n",
                "    print('speedup: %.1fx' % (f32_time / int4_time))\n",
                "except Exception as e:\n",
                "    print(f'benchmark skipped: {e}')\n",
            ]
        if "git clone" in src and "nncf" in src:
            cell["source"] = [
                "import importlib.util\n",
                "import os\n",
                "import subprocess\n",
                "import sys\n",
                "\n",
                "_deps = os.environ.get('SEMINAR_DEPS', '')\n",
                "if not importlib.util.find_spec('nncf'):\n",
                "    target = _deps or '.'\n",
                "    os.makedirs(target, exist_ok=True)\n",
                "    env = os.environ.copy()\n",
                "    env['PYTHONNOUSERSITE'] = '1'\n",
                "    subprocess.run(\n",
                "        [sys.executable, '-m', 'pip', 'install', '--target', target, 'nncf[torch]', '-q'],\n",
                "        env=env,\n",
                "        check=False,\n",
                "    )\n",
                "    if target not in sys.path:\n",
                "        sys.path.insert(0, target)\n",
                "    print('nncf installed via pip')\n",
                "else:\n",
                "    print('nncf already available')\n",
            ]
        if "pip install', '-e', f'{nncf_dir}[torch]'" in src or "pip install', '-e'" in src:
            cell["source"] = ["# nncf installed in previous cell\n", "pass\n"]
        if "torch.compile(model" in src:
            cell["source"] = [
                "import copy\n",
                "import os\n",
                "import torch\n",
                "\n",
                "class ToyLinearModel(torch.nn.Module):\n",
                "    def __init__(self, m: int, n: int, k: int):\n",
                "        super().__init__()\n",
                "        self.linear1 = torch.nn.Linear(m, n, bias=False)\n",
                "        self.linear2 = torch.nn.Linear(n, k, bias=False)\n",
                "\n",
                "    def forward(self, x):\n",
                "        x = self.linear1(x)\n",
                "        x = self.linear2(x)\n",
                "        return x\n",
                "\n",
                "model = ToyLinearModel(1024, 1024, 1024).eval().cuda()\n",
                "if os.environ.get('SEMINAR_USE_COMPILE', '0') == '1':\n",
                "    try:\n",
                "        model = torch.compile(model, mode='max-autotune', fullgraph=True)\n",
                "    except Exception as e:\n",
                "        print(f'torch.compile skipped: {e}')\n",
                "else:\n",
                "    print('torch.compile disabled for stable notebook run')\n",
                "model_f32 = copy.deepcopy(model)\n",
            ]
        if "forkflow" in src:
            cell["source"] = [s.replace("forkflow", "workflow") for s in cell["source"]]
        if "**Домашнее задание**" in src and "таблиц" in src:
            cell["source"] = [
                src.rstrip()
                + "\n\n"
                + "| Method | Size (MB) | Metric | Latency (ms) | Notes |\n"
                + "|--------|-----------|--------|--------------|-------|\n"
                + "| FP32 baseline | | | | |\n"
                + "| PTQ static (uniform qconfig) | | | | calibration subset |\n"
                + "| PTQ + sensitivity analysis | | | | per-layer qconfig |\n"
            ]

    _clear_outputs(nb)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    patch_pruning(REPO / "seminar_4" / "Pruning.ipynb")
    patch_quantization(REPO / "seminar_5" / "Post_Training_Quantization.ipynb")
    print("Patched seminar 4 and 5 notebooks for GPU.")
