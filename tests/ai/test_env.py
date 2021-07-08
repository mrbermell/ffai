import numpy as np

from tests.util import *
import pytest
import ffai.ai.env as env
import ffai.ai.env_wrappers as wrappers
import gym

from botbowlcurriculum import make_academy, all_lectures

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

    assert spat.shape == (17, 28, len(env.layers))
    assert nonspat.shape == (116,)
    assert mask.shape == (8117,)
    #assert obs[1].shape == (116, )


def test_fully_wrapped():
    env = gym.make("FFAI-wrapped-v3")

    for _ in range(10):
        spat_obs, non_spat_obs, action_mask = env.reset()
        done = False
        cum_abs_reward = 0
        reward = 0

        while not done:
            cum_abs_reward += abs(reward)
            #assert_type_and_range((spat_obs, non_spat_obs))
            action_index = np.random.choice(action_mask.nonzero()[0])
            reward, done, spat_obs, non_spat_obs, action_mask = env.step(action_index)

        assert 0 < abs(cum_abs_reward)


def test_env_reset_lecture():
    env = wrappers.GotebotWrapper(gym.make("FFAI-v3"), all_lectures)


    num_lectures = len(all_lectures)

    probs = np.random.random(num_lectures)
    probs /= sum(probs)

    levels = np.zeros(num_lectures, dtype=np.int)
    levels_and_probs = np.stack((levels, probs), axis=1)

    spat_obs, nonspat_obs, action_mask = env.reset(levels_and_probs)
    done = False

    while not done:
        action_index = np.random.choice(action_mask.nonzero()[0])
        reward, done, spat_obs, nonspat_obs, action_mask = env.step(action_index)

    env.get_lecture_outcome()


def test_env_playing_all_lectures():
    env = wrappers.GotebotWrapper(gym.make("FFAI-v3"), all_lectures)
    academy = make_academy()
    for lect_hist in academy.lect_histo:
        lect = lect_hist.lecture
        env.lecture = lect
        for level in range(lect.max_level):
            env.lecture.level = level
            spatial_obs, non_spatial_obs, action_mask = env.gen_observation(env.env_reset_with_lecture())
            assert env.env.own_team.team_id == env.env.game.state.available_actions[0].team.team_id

            done = False
            while not done:
                action_index = np.random.choice(action_mask.nonzero()[0])
                reward, done, spat_obs, nonspat_obs, action_mask = env.step(action_index)

            outcome = env.get_lecture_outcome()
            academy.log_training(outcome)
            assert len(outcome) == 3
            assert outcome[0] == all_lectures.index(lect)
            assert outcome[1] == level
            assert outcome[2] in [-1, 0, 1]

def test_wrong_action_crashes():
    env = gym.make("FFAI-wrapped-v3")
    _, _, action_mask = env.reset()

    try:
        env.step(137)
    except AssertionError:
        pass
    else:
        assert False, "Above should raise error!"


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
