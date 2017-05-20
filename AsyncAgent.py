from AbstractAgent import AbstractAgent
import torch


class AsyncAgent(AbstractAgent):

    """AsyncAgent
    """

    def __init__(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG):
        


    def observe(self):
        raise NotImplementedError

    def training(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
