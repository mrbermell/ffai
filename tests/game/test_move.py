from tests.util import *
import pytest

from pytest import set_trace


# import ffai.ai.pathfinding as pf

def assert_moves(game, player, max_dist):
    game.set_available_actions()
    avail_act = game.get_available_actions()

    positions = avail_act[0].positions
    rolls = avail_act[0].agi_rolls

    moves_left = player.num_moves_left(include_gfi=False)

    for sq, roll in zip(positions, rolls):
        dist = player.position.distance(sq)
        if dist > moves_left:  # Implying GFI
            assert len(roll) > 0

    distances = [player.position.distance(p) for p in positions]
    assert max(distances) == max_dist

def test_pathfinding_after_ball_pickup():
    game = get_game_turn()
    assert game.config.pathfinding == PathFindingOptions.SINGLE_ROLL_PATHS
    team = game.get_agent_team(game.actor)
    game.clear_board()

    player = team.players[0]
    start_square = Square(1, 1)
    game.put(player, start_square)


    target = Square(5, 4)
    game.get_ball().move_to(target)
    game.get_ball().is_carried = False

    game.set_available_actions()
    game.state.reports.clear()

    game.step(Action(ActionType.START_MOVE, player=player))
    assert_moves(game, player, player.get_ma() + 1)  # Before initial movement

    D6.fix_result(6) #successful pickup
    game.step(Action(ActionType.MOVE, position=target))


    assert player.position == target
    assert game.get_ball().is_carried
    assert not player.state.used



def test_many_moves_from_one_action():
    game = get_game_turn()
    assert game.config.pathfinding == PathFindingOptions.SINGLE_ROLL_PATHS
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()

    player = team.players[0]
    start_square = Square(1, 1)
    game.put(player, start_square)

    # opp_player = opp_team.players[0]
    # game.put(player, Square(3, 3))

    target = Square(5, 4)

    game.set_available_actions()
    game.state.reports.clear()

    game.step(Action(ActionType.START_MOVE, player=player))

    assert_moves(game, player, player.get_ma()+1 )  # Before initial movement
    game.step(Action(ActionType.MOVE, position=target))
    # assert_moves(game, player)  # After initial movement

    assert player.position == target
    assert player.state.used


def test_pathfinding_after_standing_up():
    game = get_game_turn()
    assert game.config.pathfinding == PathFindingOptions.SINGLE_ROLL_PATHS
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()

    player = team.players[0]
    start_square = Square(1, 1)
    game.put(player, start_square)
    player.state.up = False

    target = Square(5, 4)

    game.set_available_actions()
    game.state.reports.clear()

    game.step(Action(ActionType.START_MOVE, player=player))
    game.step(Action(ActionType.STAND_UP, player=player))
    assert player.state.up
    assert_moves(game, player, player.get_ma()-3+1)  # Before initial movement

    game.step(Action(ActionType.MOVE, position=target))

    assert player.position == target
    assert player.state.used


def test_pathfinding_dodging():
    game = get_game_turn()
    assert game.config.pathfinding == PathFindingOptions.SINGLE_ROLL_PATHS
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()

    player = team.players[0]
    start_square = Square(3, 3)
    game.put(player, start_square)

    opp_player = opp_team.players[0]
    game.put(opp_player, Square(2, 2))

    target = Square(6, 6)

    game.set_available_actions()
    game.state.reports.clear()

    game.step(Action(ActionType.START_MOVE, player=player))

    assert_moves(game, player, player.get_ma())  # Before initial movement

    D6.fix_result(6)  # successful dodge
    game.step(Action(ActionType.MOVE, position=target))

    assert player.position == target
    assert not player.state.used

