import math
import numpy as np

# from imitation_learning.snake import decode_state, construct_features

config_CP = {
    "name": "CartPole-v1",
    "can_render": True,
    "max_score": 500,
    "min_score": 0,
    "should_force_episode_termination_score": True,
    "should_convert_state_to_array": False,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": 0,
    "n_actions": 2,
    "actions": ["left", "right"],
    "n_attributes": 4,              
    "attributes": [
        ("Cart Position", "continuous", [-4.8, 4.8]),
        ("Cart Velocity", "continuous", [-math.inf, math.inf]),
        ("Pole Angle", "continuous", [-0.418, 0.418]),
        ("Pole Angular Velocity", "continuous", [-math.inf, math.inf])],
}

config_MC = {
    "name": "MountainCar-v0",
    "can_render": True,
    "max_score": 0,
    "min_score": -200,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": False,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": 0,
    "n_actions": 3,
    "actions": ["left", "nop", "right"],
    "n_attributes": 2,              
    "attributes": [
        ("Car Position", "continuous", [-1.2, 0.6]),
        ("Car Velocity", "continuous", [-0.07, 0.07])],
}

config_LL = {
    "name": "LunarLander-v2",
    "can_render": True,
    "n_actions": 4,
    "max_score": 1000,
    "min_score": -10000,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": False,
    "episode_termination_score": 0,
    "actions": ["nop", "left engine", "main engine", "right engine"],
    "n_attributes": 8,              
    "attributes": [
        ("X Position", "continuous", [-1.5, 1.5]),
        ("Y Position", "continuous", [-1.5, 1.5]),
        ("X Velocity", "continuous", [-5.0, 5.0]),
        ("Y Velocity", "continuous", [-5.0, 5.0]),
        ("Angle", "continuous", [-math.pi, math.pi]),
        ("Angular Velocity", "continuous", [-5.0, 5.0]),
        ("Leg 1 is Touching", "binary", [0, 1]),
        ("Leg 2 is Touching", "binary", [0, 1])],
}

config_BJ = {
    "name": "Blackjack-v0",
    "can_render": False,
    "max_score": 1,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": True,
    "conversion_fn": lambda a,b,c : c,
    "episode_termination_score": None,
    "n_actions": 2,
    "actions": ["stick", "hit"],
    "n_attributes": 3,
    "attributes": [
        ("Player's Sum", "discrete", 0, 22),
        ("Dealer's Card", "discrete", 1, 11),
        ("Usable Ace", "binary", -1, -1)],
}

config_SN = {
    "name": "Snake-8x8-v0",
    "can_render": True,
    "render_delay_ms": 100,
    "max_score": 16,
    "should_force_episode_termination_score": False,
    "should_convert_state_to_array": True,
    "conversion_fn": lambda env, s1, s2 : construct_features(env, decode_state(s1), decode_state(s2)),
    "episode_termination_score": None,
    "n_actions": 3,
    "actions": ["forward", "left", "right"],
    # "n_attributes": 36,
    "n_attributes": 7,
    "attributes": [
        ("Player's Sum", "discrete", 0, 22),
        ("Dealer's Card", "discrete", 1, 11),
        ("Usable Ace", "binary", -1, -1)],
}

def get_config(task_name):
    if task_name == "cartpole":
        return config_CP
    elif task_name == "mountain_car":
        return config_MC
    elif task_name == "lunar_lander":
        return config_LL
    elif task_name == "blackjack":
        return config_BJ
    elif task_name == "snake":
        return config_SN
        
    print(f"Invalid task_name {task_name}.")
    return None