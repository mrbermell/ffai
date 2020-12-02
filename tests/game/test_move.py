from tests.util import *
import pytest

from pytest import set_trace 
#import ffai.ai.pathfinding as pf 

def assert_moves(game, player): 
    game.set_available_actions()
    avail_act = game.get_available_actions() 
    
    positions = avail_act[0].positions 
    rolls = avail_act[0].agi_rolls  
    
    moves_left = player.num_moves_left(include_gfi=False) 
    
    for sq, roll in zip( positions, rolls ): 
        dist = player.position.distance(sq) 
        if dist > moves_left: #Implying GFI 
            assert len(roll) > 0 
            

def test_many_moves_from_one_action(): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    
    player = team.players[0]
    start_square = Square(1,1)
    game.put(player, start_square ) 
    
    target = Square(5,4)
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    game.step(Action(ActionType.START_MOVE, player=player))
    
    assert_moves(game, player) #Before initial movement 
    game.step(Action(ActionType.MOVE, position=target )) 
    assert_moves(game, player) #After initial movement 
    
    assert player.position == target 
    