# early stop
# author: Linna Fan
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_time = None
        self.early_stop = False
        
    def step(self, acc, time):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_time = time
            # self.save_checkpoint(model)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_time = time
            # self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    # def save_checkpoint(self, model):
    #     '''Saves model when validation loss decrease.'''
    #     torch.save(model.state_dict(), 'es_checkpoint_{}.pt'.format(self.model_name))
