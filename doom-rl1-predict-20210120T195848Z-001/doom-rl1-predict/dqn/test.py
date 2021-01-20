from vizdoom import *


config = "../scenarios/defend_the_center.cfg"
scene = "../scenarios/defend_the_center.wad"

game = DoomGame()
game.load_config(config)
game.set_doom_scenario_path(scene)
game.init()
game.new_episode()



moves = game.get_available_buttons()
variables = game.get_available_game_variables()
game_state = game.get_state()
action_size = game.get_available_buttons_size()
print(f"Variables: {variables}, Moves: {moves}")