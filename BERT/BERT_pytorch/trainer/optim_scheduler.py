'''A wrapper class for optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5) # This is often used in Transformer architectures, where the learning rate is inversely proportional to the square root of the model's hidden dimension.

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
    
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate() #  adjust the learning rate.
        self._optimizer.step() # updates the model parameters using the computed gradients.

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad() # Resets gradients to zero before the next forward and backward pass.
        #  In PyTorch, gradients accumulate by default, so this prevents unwanted accumulation between steps.