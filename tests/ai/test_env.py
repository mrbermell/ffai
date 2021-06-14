import numpy as np

from tests.util import *
import pytest
import ffai.ai.env as env
import ffai.ai.env_wrappers as wrappers
import gym


def test_action_wrapper():
    env = gym.make("FFAI-v3")
    env = wrappers.GotebotWrapper(env)
    env.reset()

    non_spatial_action_types = env.simple_action_types + env.defensive_formation_action_types + \
                   env.offensive_formation_action_types

    done = False
    while not done:
        action_mask = env.compute_action_masks()
        action_index = np.random.choice(action_mask.nonzero()[0])
        action_type, x, y = env.compute_action(action_index)

        assert action_type in env.available_action_types()
        if action_type in env.positional_action_types:
            assert Square(x, y) in env.available_positions(action_type)
        else:
            assert action_type in non_spatial_action_types

        _, done, _, _, _ = env.step(action_index)


def test_observation_wrapper():
    env = gym.make("FFAI-v3")
    env = wrappers.GotebotWrapper(env)
    spat, nonspat, mask = env.reset()

    assert spat.shape == (len(env.layers), 17, 28)
    assert nonspat.shape == (116,)
    assert mask.shape == (8117,)
    #assert obs[1].shape == (116, )


def test_fully_wrapped():
    env = gym.make("FFAI-wrapped-v3")

    for _ in range(10):
        spat_obs, nonspat_obs, action_mask = env.reset()
        done = False
        cum_abs_reward = 0
        reward = 0

        while not done:
            cum_abs_reward += abs(reward)
            #assert_type_and_range((spat_obs, nonspat_obs))
            action_index = np.random.choice(action_mask.nonzero()[0])
            reward, done, spat_obs, nonspat_obs, action_mask = env.step(action_index)

        assert 0 < abs(cum_abs_reward)


def assert_type_and_range(obs):
    for i, array in enumerate(obs):
        assert array.dtype == np.float32, f"obs[{i}] is not float"

        max_val = np.max(array)
        assert max_val <= 1.0, f"obs[{i}][{find_first_index(array, max_val)}] is too high ({max_val})"
        min_val = np.min(array)
        assert min_val >= 0.0, f"obs[{i}][{find_first_index(array, min_val)}] is too low ({min_val})"


def find_first_index(array: np.ndarray, value):
    indices = (array == value).nonzero()
    return [x[0] for x in indices]
