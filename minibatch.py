from __future__ import annotations

import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ExperienceReplayMemory:
    pass


class TransitionMemory(ExperienceReplayMemory):
    """
    Transition Memory untuk menyimpan sequence step sebelum disinkronisasikan ke global actor critic
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.__init__()


if __name__ == "__name__":
    raise NotImplementedError("This module is not meant to be executed directly")
