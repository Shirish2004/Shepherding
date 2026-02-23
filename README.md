# LLM-Augmented StringNet Shepherding (shepherd_codex)

This repository includes `shepherd_codex/`, a modular Python implementation of an LLM-augmented StringNet-style herding system with online adapter updates when planner behavior fails.

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q shepherd_codex/tests
```

## Run demo

```bash
python shepherd_codex/demos/demo_run.py
```

The demo prints planner intent per step and triggers online adapter fine-tuning with pre/post intent logs on detected failures.

## Run real-time PyQt viewer

```bash
python shepherd_codex/demos/live_viewer.py
```

The viewer shows sheep/dog motion in real time, the exact LLM prompt text produced by `LLMPlanner.build_prompt(...)`, decoded planner intent, and live adapter fine-tune logs.

## Run evaluation

```bash
python -c "from shepherd_codex.planner.llm_planner import LLMPlanner; from shepherd_codex.metrics.evaluation import run_evaluation; print(run_evaluation(LLMPlanner())[0])"
```

Outputs CSV and a simple plot at `shepherd_codex/eval_metrics.csv` and `shepherd_codex/eval_metrics.png`.

## Implemented modules

- `shepherd_codex/shepherd_env/env.py`: Gym-like environment (`reset`, `step`), left-quadrant sheep spawn and right-goal region.
- `shepherd_codex/shepherd_env/dynamics.py`: damped double-integrator with semi-implicit Euler.
- `shepherd_codex/shepherd_env/controllers.py`: seeking/enclosing/herding wrappers with bounded controls and StringNet-like `sig_alpha`, saturation `Ω`.
- `shepherd_codex/shepherd_env/sensors.py`: LiDAR helper and deterministic symbolic token extractor.
- `shepherd_codex/planner/llm_planner.py`: frozen mock base model + trainable adapter MLP with explicit LLM prompt design, snapshot/logging, and online update APIs.
- `shepherd_codex/planner/mock_llm.py`: deterministic base planner and oracle corrective planner.
- `shepherd_codex/metrics/failure_detector.py`: containment, formation error, and failure trigger logic.
- `shepherd_codex/metrics/evaluation.py`: scenario sweeps (`3/2`, `5/3`, `8/5`), CSV and plot outputs.
- `shepherd_codex/demos/live_viewer.py`: real-time PyQt visualizer with planner output + prompt inspector.
