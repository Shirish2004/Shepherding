"""Mock LLM planner tests on sample scene tokens."""

import numpy as np

from shepherd_env.llm import MockLLMPlanner, intent_to_embedding, scene_tokens


def test_mock_llm_produces_orient_or_split() -> None:
    planner = MockLLMPlanner()
    sheep_pos = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    sheep_vel = np.zeros_like(sheep_pos)
    tokens = scene_tokens(sheep_pos, sheep_vel, (10.0, 10.0))
    intent, conf = planner.plan(tokens)
    assert isinstance(intent, str)
    assert 0.0 <= conf <= 1.0


def test_intent_embedding_shape() -> None:
    emb = intent_to_embedding("ORIENT:0.2")
    assert emb.shape == (6,)
    assert emb.sum() == 1.0
