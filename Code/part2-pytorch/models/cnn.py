import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 7, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 
            nn.Flatten(),
            nn.Linear(32*13*13, 10),
            # nn.Softmax(dim=-1)
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        out = self.seq(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return out