# HOWTO

## Run one complete training + evaluation cycle

1. Run tests.

```bash
pytest -q
```

2. Optional warm-start via behavior cloning.

```bash
python train/bc_pretrain.py --synthetic-samples 3000 --out artifacts/bc_actor.npz
```

3. Run scratch RL training loop with curriculum.

```bash
python train/mappo_train.py --iters 20 --steps 250 --curriculum --out artifacts/train_history.json
```

4. Run evaluation grid and export metrics.

```bash
python eval/run_eval.py --episodes 5 --out artifacts/eval_metrics.csv
```

5. Plot key curves from metrics.

```python
import pandas as pd

df = pd.read_csv("artifacts/eval_metrics.csv")
agg = df.groupby(["fov", "n_rays"])["success"].mean().reset_index()
print(agg)
```

## Notes

The RL script does not require baseline demonstration files and learns from environment rewards directly. BC is a convenience warm-start only.

## Reproduce visibility ablation

Run `tests/test_scenarios.py::test_visibility_ablation_fov_changes_observability` to verify FOV-driven observability changes.
