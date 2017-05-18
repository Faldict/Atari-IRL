import torch
import torch.nn as nn


class Model:

    """ Network Model
    recurrent: bool - use recurrent connections
    histLen:   int  - number of consecutive states processed/used for backpropagation-thorough-time
    stateSpec: np   - env.observation_space
    """

    def __init__(self, recurrent, histLen, stateSpec):
        self.recurrent = recurrent
        self.histLen   = histLen
        self.stateSpec = stateSpec

    def createBody(self):
        histLen = self.recurrent and 1 or self.histLen
        net = nn.Sequential(
            nn.View(histLen * self.stateSpec[2][1], self.stateSpec[2][2], self.stateSpec[2][3]),
            nn.SpatialConvolution(histLen * self.stateSpec[2][1], 16, 8, 8, 4, 4, 1, 1),
            nn.ReLU(True),
            nn.SpatialConvolution(16, 32, 4, 4, 2, 2),
            nn.ReLU(True),
        )
        return net
