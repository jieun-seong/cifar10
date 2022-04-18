import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x): 
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape
        C_out = self.out_channels
        H_out = int((H+2*self.padding-self.kernel_size)/self.stride+1)
        W_out = int((W+2*self.padding-self.kernel_size)/self.stride+1)
        out = np.zeros((N,C_out,H_out,W_out))

        for i in range(N):
            for curr_out_ch in range(self.out_channels):
                curr_ch_bias = self.bias[curr_out_ch]
                padded = np.pad(x[i], pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding)))
                _, n_rows_padded, n_cols_padded = padded.shape
                filter = self.weight[curr_out_ch]
                j = 0
                curr_row_out = 0
                while j + self.kernel_size <= n_rows_padded:
                    k = 0
                    curr_col_out = 0
                    while k + self.kernel_size <= n_cols_padded:
                        block = padded[:,j:j+self.kernel_size,k:k+self.kernel_size]
                        out[i,curr_out_ch,curr_row_out,curr_col_out] = np.sum(filter*block) + curr_ch_bias
                        k += self.stride
                        curr_col_out += 1
                    j += self.stride
                    curr_row_out += 1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: 
        :update: dx, dw, db
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
        # Output width = (Output width + padding width right + padding width left - kernel width) / (stride width) + 1
        

        N,_,h_x,w_x = x.shape
        
        # new_padding = int((h_x-1) * self.stride / 4)
        new_padding = int(((h_x-1)*self.stride + self.kernel_size - h_x)/2)

        self.dx = np.zeros(x.shape)
        for i in range(N):
            for j in range(self.out_channels):
                dout[i,j,:,:]
                for k in range(self.in_channels):
                    filter = np.flipud(np.fliplr(self.weight[j,k,:,:]))
                    padded = np.pad(dout[i,j,:,:],new_padding)
                    h_padded, w_padded = padded.shape
                    l = 0
                    curr_row_dx = 0
                    while l + self.kernel_size <= h_padded:
                        m = 0
                        curr_col_dx = 0
                        while m + self.kernel_size <= w_padded:
                            block = padded[l:l+self.kernel_size, m:m+self.kernel_size]
                            # print("x shape", x.shape)
                            # print('w shape', self.weight.shape)
                            # print("dout shape", dout.shape)
                            # print("block shape", block.shape)
                            # print("filter shape", filter.shape)
                            # print("padded shape", padded.shape)
                            # print("kernel size", self.kernel_size)
                            # print("stride size", self.stride)
                            # print("new padding size", new_padding)
                            # print("(",i,",",k,",",curr_row_dx,",",curr_col_dx,")")
                            self.dx[i,k,curr_row_dx,curr_col_dx] += np.sum(filter*block)
                            m += self.stride
                            curr_col_dx += 1
                        curr_row_dx += 1
                        l += self.stride
        
        self.dw = np.zeros(self.weight.shape)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                filter = dout[:,i,:,:]
                curr_x = x[:,j,:,:]
                curr_x_padded = np.pad(curr_x, pad_width=((0,0),(self.padding,self.padding),(self.padding,self.padding)))
                _, h_x_padded, w_x_padded = curr_x_padded.shape
                _, h_filter, w_filter = filter.shape
                k = 0
                curr_row_dw = 0
                while k + h_filter <= h_x_padded:
                    l = 0
                    curr_col_dw = 0
                    while l + w_filter <= w_x_padded:
                        block = curr_x_padded[:,k:k+h_filter,l:l+w_filter]
                        self.dw[i,j,curr_row_dw,curr_col_dw] = np.sum(block*filter)
                        l += self.stride
                        curr_col_dw += 1
                    k += self.stride
                    curr_row_dw += 1

        self.db = np.zeros(self.bias.shape)
        for i in range(self.out_channels):
            self.db[i] = np.sum(dout[:,i,:,:])

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################