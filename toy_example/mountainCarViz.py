import gymnasium as gym
import pygame
from mountainCar_agent import game_agent

env = gym.make("MountainCar-v0", render_mode="human")

ctrl_example = [0, 2, 0, 2]


def stripe_info(state):
    location = state[0][0].item()
    velocity = state[0][1].item()
    return location, velocity


count = 0
state = env.reset()
end = False
# action = env.action_space.sample()
action = game_agent(stripe_info(state), ctrl_example)

while not end:
    ret = env.step(action)
    env.render()
    action = game_agent(stripe_info(ret), ctrl_example)
    end = ret[2]
    count += 1
env.close()
