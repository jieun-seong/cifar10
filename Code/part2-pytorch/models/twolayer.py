import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.seq = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(input_dim, hidden_size), 
            nn.Sigmoid(), 
            nn.Flatten(), 
            nn.Linear(hidden_size, num_classes)
        )
        # self.linear1 = nn.Linear(input_dim, hidden_size)
        # # self.linear1 = nn.Linear(hidden_size, input_dim)
        # # self.linear1.weight = nn.Parameter(torch.zeros(input_dim, hidden_size))
        # # self.linear1.weight = nn.Parameter(torch.zeros(hidden_size, input_dim))
        # # self.linear1.bias = nn.Parameter(torch.zeros(hidden_size))

        # self.sigmoid = nn.Sigmoid()

        # self.linear2 = nn.Linear(hidden_size, num_classes)
        # # self.linear2.weight = nn.Parameter(torch.zeros(hidden_size, num_classes))
        # # self.linear2.weight = nn.Parameter(torch.zeros(num_classes, hidden_size))
        # # self.linear2.bias = nn.Parameter(torch.zeros(num_classes))

        # # self.softmax = nn.Softmax()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # out = out.view(out.size(0), -1)
        # out = self.linear1(x)
        # out = self.sigmoid(out)
        # out = out.view(out.size(0), -1)
        # out = self.linear2(out)
        # out = self.softmax(out)
        out = self.seq(x)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out