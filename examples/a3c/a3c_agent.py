import os
from typing import Optional

import gym
from ffai import FFAIEnv
from pytest import set_trace
from torch.autograd import Variable
import torch.optim as optim
from multiprocessing import Process, Pipe
from ffai.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys


# Architecture
model_name = 'FFAI-v2'
env_name = 'FFAI-v2'
model_filename = "models/" + model_name
log_filename = "logs/" + model_name + ".dat"


class CNNPolicy(nn.Module):
    """
    Same as a2c exmaple
    """

    def __init__(self, spatial_shape, non_spatial_inputs, hidden_nodes, kernels, actions):
        super(CNNPolicy, self).__init__()

        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_shape[0], out_channels=kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding=1)

        # Non-spatial input stream
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # Linear layers
        stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        stream_size += hidden_nodes
        self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(hidden_nodes, actions)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.actor.weight.data.mul_(relu_gain)
        self.critic.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through two convolutional layers
        x1 = self.conv1(spatial_input)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)

        # Concatenate the input streams
        flatten_x1 = x1.flatten(start_dim=1)

        x2 = self.linear0(non_spatial_input)
        x2 = F.relu(x2)

        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)

        # Fully-connected layers
        x3 = self.linear1(concatenated)
        x3 = F.relu(x3)
        #x2 = self.linear2(x2)
        #x2 = F.relu(x2)

        # Output streams
        value = self.critic(x3)
        actor = self.actor(x3)

        # return value, policy
        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs


class A3CAgent(Agent):

    def __init__(self, name, env_name=env_name, cnn=None, filename: Optional[str] = model_filename):
        super().__init__(name)
        self.my_team = None
        self.env = self.make_env(env_name)

        self.spatial_obs_space = self.env.observation_space.spaces['board'].shape
        self.board_dim = (self.spatial_obs_space[1], self.spatial_obs_space[2])
        self.board_squares = self.spatial_obs_space[1] * self.spatial_obs_space[2]

        self.non_spatial_obs_space = self.env.observation_space.spaces['state'].shape[0] + \
                                     self.env.observation_space.spaces['procedures'].shape[0] + \
                                     self.env.observation_space.spaces['available-action-types'].shape[0]
        self.non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
        self.num_non_spatial_action_types = len(self.non_spatial_action_types)
        self.spatial_action_types = FFAIEnv.positional_action_types
        self.num_spatial_action_types = len(self.spatial_action_types)
        self.num_spatial_actions = self.num_spatial_action_types * self.spatial_obs_space[1] * self.spatial_obs_space[2]
        self.action_space = self.num_non_spatial_action_types + self.num_spatial_actions
        self.is_home = True

        # MODEL
        if cnn is not None:
            raise NotImplementedError

        elif filename is not None
            self.policy = torch.load(filename)
            self.policy.eval()

        self.end_setup = False
        self.cnn_used = False

    def act(self, game):
        self.env.game = game
        action_object, _, _, _, _, _ = self.act_when_training(self.env)
        return action_object

    def act_when_training(self, env):
        if self.end_setup:
            self.end_setup = False

            # Todo, need to return the whole tuple here, set the action_mask to only show end_setup.
            return self.create_action_object(ActionType.END_SETUP)

        action_dict, actions, action_masks, value, spatial_obs, non_spatial_obs = self.act_NN(env)

        # Let's just end the setup right after picking a formation
        if action_dict["action-type"].name.lower().startswith('setup'):
            self.end_setup = True

        return action_dict, actions, action_masks, value, spatial_obs, non_spatial_obs

    def act_NN(self, env, obs=None):
        if obs is None:
            obs = env.get_observation()

        # Flip board observation if away team
        if not self.is_home:
            obs['board'] = self._flip(obs['board'])

        spatial_obs, non_spatial_obs = self._update_obs(obs)

        action_masks = self._compute_action_masks(obs)
        action_masks = torch.tensor(action_masks, dtype=torch.bool)

        values, actions = self.policy.act(
            Variable(spatial_obs.unsqueeze(0)),
            Variable(non_spatial_obs.unsqueeze(0)),
            Variable(action_masks.unsqueeze(0)))

        values.detach()
        # Create action from output
        action = actions[0]
        value = values[0]
        value.detach()
        action_type, x, y = self._compute_action(action.numpy()[0])

        # Flip position if playing as away and x>0 (x>0 means it's a positional action)
        if not self.is_home and x > 0:
            x = env.game.arena.width - 1 - x

        action_object = {'action-type': action_type, 'x': x, 'y': y}

        return action_object, actions, action_masks, value, spatial_obs, non_spatial_obs

    def create_action_object(self, action_dict):
        if action_dict["action-type"] is None:
            return None

        if action_dict["action-type"] in FFAIEnv.positional_action_types:
            assert action_dict["x"] is not None and action_dict["y"] is not None
            square = Square(action_dict["x"], action_dict["y"])
            return ffai.Action(action_dict["action-type"], position=square, player=None)
        else:
            return ffai.Action(action_dict["action-type"], position=None, player=None)

    def new_game(self, game, team):
        self.my_team = team
        self.is_home = self.my_team == game.state.home_team

    def _flip(self, board):
        flipped = {}
        for name, layer in board.items():
            flipped[name] = np.flip(layer, 1)
        return flipped

    def end_game(self, game):
        pass

    def _compute_action_masks(self, ob):
        mask = np.zeros(self.action_space)
        i = 0
        for action_type in self.non_spatial_action_types:
            mask[i] = ob['available-action-types'][action_type.name]
            i += 1
        for action_type in self.spatial_action_types:
            if ob['available-action-types'][action_type.name] == 0:
                mask[i:i + self.board_squares] = 0
            elif ob['available-action-types'][action_type.name] == 1:
                position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
                position_mask_flatten = np.reshape(position_mask, (1, self.board_squares))
                for j in range(self.board_squares):
                    mask[i + j] = position_mask_flatten[0][j]
            i += self.board_squares
        assert 1 in mask
        return mask

    def _compute_action(self, action_idx):
        if action_idx < len(self.non_spatial_action_types):
            return self.non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - self.num_non_spatial_action_types
        spatial_pos_idx = spatial_idx % self.board_squares
        spatial_y = int(spatial_pos_idx / self.board_dim[1])
        spatial_x = int(spatial_pos_idx % self.board_dim[1])
        spatial_action_type_idx = int(spatial_idx / self.board_squares)
        spatial_action_type = self.spatial_action_types[spatial_action_type_idx]
        return spatial_action_type, spatial_x, spatial_y

    def _update_obs(self, obs):
        """
        Takes the observation returned by the environment and transforms it to an numpy array that contains all of
        the feature layers and non-spatial info.
        """

        spatial_obs = np.stack(obs['board'].values())

        state = list(obs['state'].values())
        procedures = list(obs['procedures'].values())
        actions = list(obs['available-action-types'].values())

        non_spatial_obs = np.stack(state + procedures + actions)
        non_spatial_obs = np.expand_dims(non_spatial_obs, axis=0)

        return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()

    def make_env(self, env_name):
        env = gym.make(env_name)
        return env


# Register the bot to the framework
ffai.register_bot('a3c-agent', A3CAgent)
