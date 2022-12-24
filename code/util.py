import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
	
def check_device():
	print("### Device Check list ###")
	print("GPU available?:", torch.cuda.is_available())
	device_number = torch.cuda.current_device()
	print("Device number:", device_number)
	print("Is device?:", torch.cuda.device(device_number))
	print("Device count?:", torch.cuda.device_count())
	print("Device name?:", torch.cuda.get_device_name(device_number))
	print("### ### ### ### ### ###\n\n")

def save_model(model, step, dir):
	fname = "{:06d}_model.pt"
	torch.save(model.state_dict(), dir+fname.format(step))
	print("Model saved.")

class EarlyStopping:

  def __init__(self, patience=7, verbose=False, delta=0, path="./weights"):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta
    self.path = path
    

  def __call__(self, val_loss, model, epoch):
    
    if self.best is None:
      self.best = val_loss
      self.save_checkpoint(val_loss, model, epoch)

    elif val_loss > self.best + self.delta:
      self.counter += 1
      print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True

    else:
      if val_loss < self.best:
        self.best = val_loss
        self.save_checkpoint(val_loss, model, epoch)
      self.counter = 0

  def save_checkpoint(self, val_loss, model, epoch):
    if self.verbose:
      print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).		Saving model ...')

    path = self.path + f'/checkpoint_{val_loss:.6f}_{epoch:03d}'
    torch.save(model.state_dict(), path)
    self.val_loss_min = val_loss