import argparse
from util import *
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import Dataset
from build_model import build_model
np.set_printoptions(linewidth=np.inf)

def test():
    parser = argparse.ArgumentParser()

    ###################################################################################
    # seed: random성을 고정하는 값                                                  ###
    parser.add_argument('--seed', required=False, type=int, default=0)              ###
    # normalization: data를 모델에 넣기 전에 표준화 할건지                          ###
    parser.add_argument('--normalization', required=False, type=bool, default=True) ###
    ###################################################################################
        
    # model parameters
    # train에서 사용한 값이랑 동일하게 설정해야 합니다!
    parser.add_argument('--window_size', required=False, type=int, default=3)
    parser.add_argument('--n_layer', required=False, type=int, default=4)
    parser.add_argument('--d_model', required=False, type=int, default=512)
    parser.add_argument('--h', required=False, type=int, default=8)
    parser.add_argument('--d_ff', required=False, type=int, default=2048)
    parser.add_argument('--dr_rate', required=False, type=float, default=0.3)
    

    args = parser.parse_args()

    set_seed(args.seed)

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    path_to_train_input_data = "../datasets/test/test_saved_x_"+str(args.window_size)+ ".npy"
    path_to_train_label_data = "../datasets/test/test_saved_y_"+str(args.window_size)+ ".npy"
    input = np.load(path_to_train_input_data, allow_pickle=True)
    label = np.load(path_to_train_label_data, allow_pickle=True)
    
    ## normalization
    stats = np.load('../datasets/mean_std.npy', allow_pickle=True)
    means = stats[0]
    stds = stats[1]

    if args.normalization:
    
        input_means = np.concatenate([np.repeat(means[:-1], args.window_size), np.repeat(means[-1], args.window_size)], axis=0)
        input_stds = np.concatenate([np.repeat(stds[:-1], args.window_size), np.repeat(stds[-1], args.window_size)], axis=0)

        label_means = np.repeat(means[-1], args.window_size)
        label_stds = np.repeat(stds[-1], args.window_size)
  
        input = (input - input_means) / input_stds
        label = (label - label_means) / label_stds

    ## make dataset
    test_dataset = Dataset.FootBall_Dataset(
        input, label, device=device
    )

    print(f"Testing Data Size : {len(test_dataset)}")

    ## make data loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False)
    

    model = build_model(src_vocab_size=17, tgt_vocab_size=1, device=device, max_len=args.window_size, n_layer=args.n_layer, d_model=args.d_model, h=args.h, d_ff=args.d_ff, dr_rate=args.dr_rate, norm_eps=1e-5)
    weight_path = "./weights/" + str(args.window_size) + '/checkpoint_0.200520_037'
    model.load_state_dict(torch.load(weight_path))
    
    loss_fn = nn.MSELoss().to(device)

    start_time = time.time()
    model.eval()
    test_loss = 0.
    for idx_batch, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        
        output = output[:,-1]
        y = y[:,-1]

        if idx_batch == 0:
            test_output_pred = output.cpu().data.numpy()
        else:
            test_output_pred = np.concatenate((test_output_pred, output.cpu().data.numpy()))

        test_loss = loss_fn(output, y).item()

    elapsed_time = time.time() - start_time

    print(f'Inference Time: {elapsed_time: .3f}')
    print(f'Test Loss (MSE): {test_loss: .6f}')

    for i in range(len(label)):
        pred = test_output_pred[i]*stds[-1] + means[-1]
        y = label[i,-1]*stds[-1] + means[-1]
        
        print(f'Prediction : {pred: 8.0f}  GND truth : {y: 8.0f}')





if __name__ == '__main__':
    test()
