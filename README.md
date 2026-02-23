# LiDAR-guided Shepherding Research Scaffold

This repository provides a modular multi-agent shepherding environment with physically grounded damped double-integrator dynamics, sheep LiDAR sensing, deterministic mock LLM planning, and RL training/evaluation scaffolds.

## Main components

- `shepherd_env/` environment API and simulation modules.
- `train/` starter scripts for scratch RL, optional behavior cloning warm-start, and LLM adapter training.
- `eval/` evaluation sweeps over `{N_a, N_d, FOV, n_rays}`.
- `tests/` unit and integration tests for LiDAR, spawn, dynamics, planner, and scenarios.
- `configs/` scenario defaults.

## Environment API

```python
from shepherd_env import ShepherdEnv, EnvConfig

env = ShepherdEnv(EnvConfig())
obs = env.reset(seed=42)
obs, rewards, done, info = env.step({"dog_0": [0.0, 0.0]})
```

`ShepherdEnv` exposes:

- `reset(seed=None, config=None)`
- `step(action_dict)`
- `render(mode='human'|'rgb_array')`
- `set_spawn_config(spawn_cfg)`
- `get_state()` and `set_state(state)`

## Physics and sensing

- Dynamics: `r_dot=v`, `v_dot=u-C_D|v|v`, integrated by semi-implicit Euler.
- Sheep controller: Reynolds-like flocking plus dog repulsion.
- Dogs: continuous acceleration controls with safety projection and fallback baseline control.
- Sheep LiDAR: ray-casting against circle-bodied agents and polygon obstacles with configurable FOV in `[60, 120]`.

## RL without baseline demos

- `train/mappo_train.py` runs a scratch REINFORCE-style loop and does not need demonstration files.
- `train/bc_pretrain.py` accepts an optional dataset but can auto-generate synthetic pseudo-demos from the built-in baseline controller if none are provided.

## Quick start

```bash
python -m pytest -q
python train/mappo_train.py --iters 8 --steps 200 --curriculum --out train_history.json
python eval/run_eval.py --episodes 2 --out eval/metrics.csv
```
