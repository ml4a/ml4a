import IPython


class ProgressBar:
    
    def __init__(self, total_iter, num_increments=32):
        self.num_increments = num_increments
        self.idx_iter, self.total_iter = 0, total_iter
        self.iter_per = self.total_iter / self.num_increments

    def update(self, update_str=''):
        self.idx_iter += 1
        progress_iter = int(self.idx_iter / self.iter_per)
        progress_str  = '[' + '=' * progress_iter 
        progress_str += '-' * (self.num_increments - progress_iter) + ']'
        IPython.display.clear_output(wait=True)
        IPython.display.display(progress_str+'  '+update_str)


def log(message, verbose=True):
    if not verbose:
        return
    print(message)


def warn(condition, message, verbose=True):
    if not condition:
        return
    log('Warning: %s' % message, verbose)


    