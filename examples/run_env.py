"""Example rollout script for ShepherdEnv."""

from shepherd_env import EnvConfig, ShepherdEnv


def main() -> None:
    cfg = EnvConfig()
    env = ShepherdEnv(cfg)
    obs = env.reset(seed=1)
    for _ in range(50):
        action = {f"dog_{j}": env._baseline_dog_action(j) for j in range(cfg.n_dogs)}
        obs, rewards, done, info = env.step(action)
        if done:
            break
    env.render("human")


if __name__ == "__main__":
    main()
