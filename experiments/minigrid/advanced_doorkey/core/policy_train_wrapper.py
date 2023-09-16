from typing import Tuple
from gymnasium.core import Env, Wrapper 
from minigrid.core.world_object import Door, Key
from experiments.minigrid.utils import actions 
import numpy as np 
import matplotlib.pyplot as plt 
from collections import namedtuple
import torch

KeyTuple = namedtuple("Key", "position colour")
DoorTuple = namedtuple("Door", "position colour")

class AdvancedDoorKeyPolicyTrainWrapper(Wrapper):
    def __init__(self, 
                 env: Env,
                 check_option_complete=lambda x: False,
                 option_reward: int=1,
                 key_colours: list=[],
                 door_colour: str=None,
                 key_collected: bool=False,
                 door_unlocked: bool=False,
                 door_open: bool=False,
                 time_limit: int=2000):
        super().__init__(env)
        
        self.objects = {
            "keys": [],
            "door": None,
            "goal": None
        }
        self.door_colour = door_colour
        self.key_colours = key_colours
        self.door_key_position = None
        self._timestep =  0
        self.time_limit = time_limit
        
        self.door = None
        
        self.check_option_complete = check_option_complete
        self.option_reward = option_reward
        
        self.key_collected = key_collected
        self.door_unlocked = door_unlocked
        self.door_open = door_open
        
    
    def _modify_info_dict(self, info):
        info['timestep'] = self._timestep
        info['keys'] = self.objects["keys"]
        info['door'] = self.objects["door"]
        info['goal'] = self.objects["goal"]
        
        return info
    
    def _get_door_key(self):
        for key in self.objects['keys']:
            if key.colour == self.objects["door"].colour:
                return key
    
    def get_door_obj(self):
        door = self.env.unwrapped.grid.get(
            self.objects["door"].position[0], 
            self.objects["door"].position[1]
        )
        
        return door
    
    def reset(self, agent_reposition_attempts=0):
        
        obs, info = self.env.reset()
        
        self._find_objs()
        self._set_door_colour()
        self._set_key_colours()
        
        if self.key_collected or self.door_unlocked or self.door_open:
            correct_key = self._get_door_key()
            key = self.env.unwrapped.grid.get(correct_key.position[0], correct_key.position[1])
            self.env.unwrapped.carrying = key
            self.env.unwrapped.carrying.cur_pos = np.array([-1, -1])
            self.env.unwrapped.grid.set(correct_key.position[0],
                                        correct_key.position[1],
                                        None)
        
            if self.door_unlocked or self.door_open:
                door = self.env.unwrapped.grid.get(
                    self.objects["door"].position[0], 
                    self.objects["door"].position[1]
                )
                door.is_locked = False
            
                if self.door_open:
                    door = self.env.unwrapped.grid.get(
                        self.objects["door"].position[0], 
                        self.objects["door"].position[1]
                    )
                    door.is_locked = False
                    door.is_open = True
        
        self.env.unwrapped.place_agent_randomly(agent_reposition_attempts)
        
        obs, _, _, info = self.env.step(actions.LEFT)
        obs, _, _, info = self.env.step(actions.RIGHT)
        
        self.env.unwrapped.time_step = 0
        self._timestep = 0
        
        info = self._modify_info_dict(info)
        
        # fig = plt.figure(num=1, clear=True)
        # ax = fig.add_subplot()
        # ax.imshow(np.transpose(obs, axes=[1,2,0]))
        # plt.show(block=False)
        # input("Option completed. Continue?")
        
        if type(obs) is np.ndarray:
            obs = torch.from_numpy(obs).float()
        
        
        return obs, info
    
    
    def _find_objs(self):
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell:
                    if cell.type == "key":
                        self.objects["keys"].append(
                            KeyTuple((x, y), cell.color)
                        )
                    elif cell.type == "door":
                        self.door = cell
                        self.objects["door"] = DoorTuple((x, y), cell.color)
                    elif cell.type == "goal":
                        self.objects["goal"] = (x, y)
                    elif cell.type == "wall":
                        continue
                    else:
                        raise Exception("Unrecognized object {} found at ({},{})".format(cell, x, y))
        
        if self.door_key_position is None:
            door_key = self._get_door_key()
            self.door_key_position = door_key.position
        
    def _set_door_colour(self):
        if self.door_colour is None:
            self.door_colour = self.objects["door"].colour
            return
        
        new_door = Door(self.door_colour, is_locked=True)
        
        self.env.unwrapped.grid.set(
            self.objects["door"].position[0],
            self.objects["door"].position[1],
            new_door
        )
        old_colour = self.objects["door"].colour
        self.objects["door"] = DoorTuple(self.objects["door"].position,
                                                self.door_colour)
        
        new_key = Key(self.door_colour)
        keys = []
        for key in self.objects["keys"]:
            if key.colour == old_colour:
                self.env.unwrapped.grid.set(
                    key.position[0],
                    key.position[1],
                    new_key
                )
                keys.append(KeyTuple(key.position,
                                     self.door_colour))
            else:
                keys.append(key)
        self.objects["keys"] = keys
        
    def _set_key_colours(self):
        if len(self.key_colours) == 0:
            return
        
        c_idx = 0
        for idx, key in enumerate(self.objects["keys"]):
            if key.position == self.door_key_position:
                continue
            if c_idx > len(self.key_colours):
                return
            
            colour = self.key_colours[c_idx]
            
            new_key = Key(colour)
            self.env.unwrapped.grid.set(
                key.position[0],
                key.position[1],
                new_key
            )
            self.objects["keys"][idx] = KeyTuple(key.position,
                                                 colour)
            c_idx += 1
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._timestep += 1
        info = self._modify_info_dict(info)
        if self._timestep >= self.time_limit:
            done = True
        if np.max(obs) > 1:
            obs = obs/255
        if type(obs) is np.ndarray:
            obs = torch.from_numpy(obs).float()
        if self.check_option_complete(self):
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # screen = self.env.render()
            # ax.imshow(screen)
            # plt.show(block=False)
            # input("Option completed. Continue?")
            return obs, 1, True, info
        else:
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # screen = self.env.render()
            # ax.imshow(screen)
            # plt.show(block=False)
            # input("continue?")
            return obs, 0, done, info
        
        
        
        
        
        



