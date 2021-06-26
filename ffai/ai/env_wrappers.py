from abc import ABC

from pytest import set_trace

from ffai.core.model import OutcomeType
from ffai.ai.env import FFAIEnv
import gym
import numpy as np


def make_wrapped_env(**kwargs):
    env = gym.make("FFAI-v3")
    env = GotebotWrapper(env)
    return env

rewards = { # Negative for opp
    OutcomeType.TOUCHDOWN: 1,
    OutcomeType.SUCCESSFUL_CATCH: 0.1,
    OutcomeType.INTERCEPTION: 0.2,
    OutcomeType.SUCCESSFUL_PICKUP: 0.1,
    OutcomeType.FUMBLE: -0.1,
    OutcomeType.KNOCKED_DOWN: -0.1,
    OutcomeType.KNOCKED_OUT: -0.2,
    OutcomeType.CASUALTY: -0.5
}


class MaskStupidActions(gym.Wrapper, ABC):
    pass # TODO: START_PASS, START_HANDOFF, START_FOUL, PASS,


class GotebotWrapper(gym.Wrapper, ABC):
    def __init__(self, env):
        super().__init__(env)

        self._x_max = env.action_space["x"].n
        self._y_max = env.action_space["y"].n
        self._board_squares = self._x_max * self._y_max

        self._non_spatial_action_types = env.simple_action_types + \
                                         env.defensive_formation_action_types + \
                                         env.offensive_formation_action_types

        self._num_spat_actions = len(self.env.positional_action_types)
        self._num_non_spat_action = len(self._non_spatial_action_types)

        self.action_space = gym.spaces.Discrete(
            self._num_non_spat_action + self._board_squares * self._num_spat_actions)

        self.action_mask = None

        # non_spat_keys = ['state', 'procedures', 'available-action-types']
        # num_non_spat_obs = sum([env.observation_space[s].shape[0] for s in non_spat_keys])

        self.observation_space = env.observation_space['board']

        self._spat_shape = None
        self._non_spat_shape = None
        self._action_shape = None

    def step(self, action):
        observation, reward, done, info = self.env.step(self.convert_action(action)) # FFAIEnv return
        if not done:
            spat_obs, nonspat_obs, action_mask = self.gen_observation(observation)
        else:
            spat_obs = np.zeros(self._spat_shape, dtype=np.float32)
            nonspat_obs = np.zeros(self._non_spat_shape, dtype=np.float32)
            action_mask = np.zeros(self._action_shape, dtype=np.bool)

        return self.reward(reward), done, spat_obs, nonspat_obs, action_mask

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        spatial_obs, non_spatial_obs, action_mask = self.gen_observation(obs)

        self._spat_shape = spatial_obs.shape
        self._non_spat_shape = non_spatial_obs.shape
        self._action_shape = action_mask.shape

        return spatial_obs, non_spatial_obs, action_mask

    def convert_action(self, action):
        action_type, x, y = self.compute_action(action)

        return {'action-type': action_type,
                'x': x,
                'y': y}

    def compute_action(self, action_idx):

        assert action_idx in self.action_mask.nonzero()[0], f"Action {action_idx} is not in action mask!"
        #if action_idx not in self.action_mask.nonzero()[0]:
        #    action_idx = np.where(self.action_mask)[0][0]

        if action_idx < self._num_non_spat_action:
            return self._non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - self._num_non_spat_action
        spatial_pos_idx = spatial_idx % self._board_squares
        spatial_y = int(spatial_pos_idx / self._x_max)
        spatial_x = int(spatial_pos_idx % self._x_max)
        spatial_action_type_idx = int(spatial_idx / self._board_squares)
        spatial_action_type = self.env.positional_action_types[spatial_action_type_idx]

        return spatial_action_type, spatial_x, spatial_y

    def compute_action_masks(self):

        mask = np.zeros(self.action_space.n, dtype=bool)

        for action_type in self.env.available_action_types():
            if action_type in self._non_spatial_action_types:
                index = self._non_spatial_action_types.index(action_type)
                mask[index] = True
            elif action_type in self.env.positional_action_types:
                action_start_index = self._num_non_spat_action + \
                                     self.env.positional_action_types.index(action_type) * self._board_squares

                for pos in self.env.available_positions(action_type):
                    mask[action_start_index + pos.x + pos.y * self._x_max] = True

        assert True in mask, "No available action in action_mask"
        self.action_mask = mask
        return mask


    def gen_observation(self, obs):
        spatial_obs = np.transpose(np.array(list(obs['board'].values()), dtype=np.float32), (1,2,0) )

        non_spatial_obs = np.array(list(obs['state'].values()) +
                                   list(obs['procedures'].values()) +
                                   list(obs['available-action-types'].values()), dtype=np.float32)

        # non_spatial_obs = np.expand_dims(non_spatial_obs, axis=0)

        return spatial_obs, non_spatial_obs, self.compute_action_masks()

    def reward(self, reward=0.0):
        r = reward
        for outcome in self.env.get_outcomes():
            team = None
            if outcome.player is not None:
                team = outcome.player.team
            elif outcome.team is not None:
                team = outcome.team
            if outcome.outcome_type in rewards:
                reward = rewards[outcome.outcome_type]

                if team == self.env.own_team:
                    r += reward
                elif team == self.env.opp_team:
                    r -= reward
                    # if info['ball_progression'] > 0:
        #    r += info['ball_progression'] * ball_progression_reward
        return r
