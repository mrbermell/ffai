from tests.util import *
import pytest

from pytest import set_trace 
#import ffai.ai.pathfinding as pf 

def test_many_moves_from_one_action(): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    
    player = team.players[0]
    game.put(player, Square(1,1) ) 
    
    target = Square(5,4)
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    game.step(Action(ActionType.START_MOVE, player=player))
    game.step(Action(ActionType.MOVE, position=target )) 
    
    assert player.position == target 
    