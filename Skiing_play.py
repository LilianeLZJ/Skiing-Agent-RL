class my_class(object):
    pass

import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.wrappers import RecordVideo
import ale_py

# check whether skiing environment is available
# print(gym.envs.registry.keys())

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

# create skiing environment
# the rendering mode should be "rgb_array" to play the game
env = gym.make('ALE/Skiing-v5', render_mode="rgb_array")
# wrap the environment with RecordVideo. The episode_trigger lambda here records every episode.
env = RecordVideo(env, video_folder="human player_videos/", episode_trigger=lambda episode_id: True)
print(env.action_space)
# discrete(3): nood:0, left:2, right:1

#using keyboard control
play(env,keys_to_action ={"w":0,"a":2,"d":1})
