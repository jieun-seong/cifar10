import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W = x.shape
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
        # Output width = (Output width + padding width right + padding width left - kernel width) / (stride width) + 1
        H_out = int((H-self.kernel_size)/self.stride+1)
        W_out = int((W-self.kernel_size)/self.stride+1)
        out = np.zeros((N, C, H_out, W_out))
        curr_r_idx = 0
        curr_c_idx = 0
        for i in range(N):
            for j in range(C):
                k = 0
                curr_r_idx = 0
                while k < H:
                    l = 0
                    curr_c_idx = 0
                    while l < W:
                        block = x[i,j,k:k+self.kernel_size,l:l+self.kernel_size]
                        out[i,j,curr_r_idx,curr_c_idx] = block.max()
                        l += self.stride
                        curr_c_idx += 1
                    k += self.stride
                    curr_r_idx += 1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        :modify: self.dx
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape
        _, _, H_dout, W_dout = dout.shape
        self.dx = np.zeros(x.shape)
        for i in range(N):
            for j in range(C):
                k = 0
                while k < H:
                    l = 0
                    while l < W:
                        block = x[i,j,k:k+self.kernel_size,l:l+self.kernel_size]
                        block_max = block.max()
                        for k2 in range(self.kernel_size):
                            for l2 in range(self.kernel_size):
                                if block[k2,l2] == block_max:
                                    curr_k = k+k2
                                    curr_l = l+l2
                                    # self.dx[i,j,k+k2,l+l2] = 1 
                                    self.dx[i,j,curr_k,curr_l] = dout[i,j,int(curr_k/self.kernel_size), int(curr_l/self.kernel_size)]
                        l += self.stride
                    k += self.stride
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
