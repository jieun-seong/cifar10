import torch
import torch.nn as nn

# You will re-use the contents of this file for your eval-ai submission.

class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 4, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 
            nn.Flatten(),
            nn.Linear(12544, 512),
        )

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        outs = self.seq(x)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs