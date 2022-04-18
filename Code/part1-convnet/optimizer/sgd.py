from sre_constants import LITERAL_UNI_IGNORE
from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight
        

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                v_w = self.grad_tracker[idx]['dw']
                v_w = v_w * self.momentum - self.learning_rate * m.dw
                self.grad_tracker[idx]['dw']  = v_w
                m.weight += v_w
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                v_b = self.grad_tracker[idx]['db']
                v_b = v_b * self.momentum - self.learning_rate * m.db
                self.grad_tracker[idx]['db']  = v_b
                m.bias += v_b
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################