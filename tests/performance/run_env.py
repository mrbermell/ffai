#!/usr/bin/env python3

import gym
import numpy as np
import ffai
from ffai.core.game import Game
from ffai.core.model import Agent
import ffai.ai.env_wrappers as wrappers

#import cProfile
#import io
#import pstats

import time

def profile_and_print_result(function, sortkey="tottime"):
    """
    Choose sortkey from: 'ncalls', 'tottime', 'percall', 'cumtime', 'percall', and others
    """
    pr = cProfile.Profile()

    pr.enable()
    function()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sortkey)
    ps.print_stats(50)
    print(s.getvalue())


def run_env(n, env_name="FFAI-v3", enable_forward_model=False, pathfinding=False):
    env = gym.make(env_name)
    env.config.pathfinding_enabled = pathfinding
    env = wrappers.GotebotWrapper(env)

    for i in range(n):
        seed = i
        env.env.seed(seed)
        rnd = np.random.RandomState(seed)
        np.random.seed(seed)

        _, _, action_mask = env.reset()
        if enable_forward_model:
            env.env.game.enable_forward_model()

        done = False

        while not done:
            action_index = np.random.choice(action_mask.nonzero()[0], )

            _, done, _, _, action_mask = env.step(action_index)


def get_random_action_from_env(env, random_state):
    action_types = env.available_action_types()
    action_type = random_state.choice(action_types)

    available_positions = env.available_positions(action_type)
    pos = random_state.choice(available_positions) if len(available_positions) > 0 else None

    return {'action-type': action_type,
            'x': pos.x if pos is not None else None,
            'y': pos.y if pos is not None else None}


def run_game(nbr_of_games, enable_forward_model):
    config = ffai.load_config("gym-11")
    config.fast_mode = True
    ruleset = ffai.load_rule_set(config.ruleset)
    home = ffai.load_team_by_filename("human", ruleset)
    away = ffai.load_team_by_filename("human", ruleset)
    away_agent = Agent("Human 1", human=True, agent_id=1)
    home_agent = Agent("Human 2", human=True, agent_id=2)

    seed = 0
    random_agent = ffai.make_bot('random')
    random_agent.rnd = np.random.RandomState(seed)

    for _ in range(nbr_of_games):
        game = Game(seed, home, away, home_agent, away_agent, config)
        game.init()
        if enable_forward_model:
            game.enable_forward_model()
        while not game.state.game_over:
            game.step(random_agent.act(game))


if __name__ == "__main__":
    nbr_of_games = 3



    print(f"---- Playing {nbr_of_games} games - pathfinding on ------")

    start = time.time()
    run_env(nbr_of_games, env_name="FFAI-v3", enable_forward_model=False, pathfinding=True)
    end = time.time()

    print(f"Played {nbr_of_games} in {end-start:.2f} seconds")

    #profile_and_print_result(function=lambda: run_env(nbr_of_games, env_name="FFAI-v3", enable_forward_model=False, pathfinding=True), sortkey="tottime")

    #print(f"---- Game played {nbr_of_games} times - no pathfinding ------")
    #profile_and_print_result(function=lambda: run_env(nbr_of_games, env_name="FFAI-v3", enable_forward_model=False, pathfinding=False), sortkey="tottime")


    #print(f"---- Game played {nbr_of_games} times - forward model enabled ------")
    #profile_and_print_result(function=lambda: run_game(nbr_of_games, enable_forward_model=True), sortkey="tottime")

    #print(f"---- Gym played {nbr_of_games} times - forward model disabled ------")
    #profile_and_print_result(function=lambda: run_env(nbr_of_games, enable_forward_model=False), sortkey="tottime")



