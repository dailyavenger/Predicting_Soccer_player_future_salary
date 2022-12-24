import argparse
from util import *
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import Dataset
from tqdm import tqdm
from build_model import build_model

def train():
    parser = argparse.ArgumentParser()

    ###################################################################################
    # seed: random성을 고정하는 값                                                  ###
    parser.add_argument('--seed', required=False, type=int, default=0)              ###
    # normalization: data를 모델에 넣기 전에 표준화 할건지                          ###
    parser.add_argument('--normalization', required=False, type=bool, default=True) ###
    ###################################################################################
    
    # not model parameters
    # epoch: 학습할 횟수
    parser.add_argument('--epoch', required=False, type=int, default=100)
    # batch size: 한 batch에 몇개 데이터 넣을건지
    parser.add_argument('--batch_size', required=False, type=int, default=128)
    # initial learning rate: 학습을 시작할 때의 learning rate
    parser.add_argument('--initial_lr', required=False, type=float, default=0.001)
    # gamma: learning rate에 곱해져서 learning rate를 감소시키는 정도
    parser.add_argument('--gamma', required=False, type=float, default=0.99)
    # early stopping: valid loss를 보고 학습을 멈출 수 있도록 하는 인자
    parser.add_argument('--early_stopping', required=False, type=bool, default=True)
    # patience: early stopping을 사용할 때 valid loss가 값이 낮아지지 않더라도 몇번의 학습을 더 지켜보고 멈출건지
    parser.add_argument('--patience', required=False, type=int, default=10)
    # delta: earlystopping을 사용할 때 valid loss < best valid loss + delta 라면 더 작다고 판단
    parser.add_argument('--delta', required=False, type=float, default=0.08)
    
    # model parameters
    # window_size: 몇년치 정보를 볼건지 (3,4,5,6,7) 중 한 값
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
    check_device()

    print("========== Loading Dataset =========")
    path_to_train_input_data = "../datasets/train/train_saved_x_"+str(args.window_size)+ ".npy"
    path_to_train_label_data = "../datasets/train/train_saved_y_"+str(args.window_size)+ ".npy"
    input = np.load(path_to_train_input_data, allow_pickle=True)
    label = np.load(path_to_train_label_data, allow_pickle=True)


    print("=========== Data Preprocessing =========== ")
    
    ## normalization
    if args.normalization:
        stats = np.load('../datasets/mean_std.npy', allow_pickle=True)
        means = stats[0]
        stds = stats[1]
    
        input_means = np.concatenate([np.repeat(means[:-1], args.window_size), np.repeat(means[-1], args.window_size)], axis=0)
        input_stds = np.concatenate([np.repeat(stds[:-1], args.window_size), np.repeat(stds[-1], args.window_size)], axis=0)

        label_means = np.repeat(means[-1], args.window_size)
        label_stds = np.repeat(stds[-1], args.window_size)
  
        input = (input - input_means) / input_stds
        label = (label - label_means) / label_stds

    ## make dataset
    dataset = Dataset.FootBall_Dataset(
        input, label, device=device
    )

    ## train / val split
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(val_dataset)}")


    print("Putting data to loader...", end="")
    ## make data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print("completed")
    
    print("Loading model/optim/scheduler...", end="")


    model = build_model(src_vocab_size=17, tgt_vocab_size=1, device=device, max_len=args.window_size, n_layer=args.n_layer, d_model=args.d_model, h=args.h, d_ff=args.d_ff, dr_rate=args.dr_rate, norm_eps=1e-5)
    
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.initial_lr, weight_decay=0)
    lr_step_size = int(len(train_dataset)/args.batch_size)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=args.gamma)

    weight_path = "./weights/" + str(args.window_size)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=args.delta, path=weight_path)

    print("completed")

    train_loss_history = []
    val_loss_history = []

    print("########## Start Train ##########")

    for idx_epoch in range(args.epoch):
        start_time = time.time()

        model.train()
        train_loss = 0.
        for idx_batch, (x, y) in enumerate(train_loader):
            model.zero_grad()

            x, y = x.to(device), y.to(device)
            output = model(x)

            # loss 구할 때 마지막 연봉만 비교하고 싶으면, 이 코드를 사용하면 됩니다.
            # output = output[:,-1]
            # y = y[:,-1]

            loss = loss_fn(output, y)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_sch.step()
            
        train_loss /= idx_batch+1
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.
        for idx_batch, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            
            # loss 구할 때 마지막 연봉만 비교하고 싶으면, 이 코드를 사용하면 됩니다.
            # output = output[:,-1]
            # y = y[:,-1]

            loss = loss_fn(output, y)

            val_loss += loss.item()

        val_loss /= idx_batch+1
        val_loss_history.append(val_loss)

        elapsed_time = time.time() - start_time

        print("\r %05d | Train Loss: %.8f | Valid Loss: %.8f | lr: %.7f | time: %.3f" % (idx_epoch+1, train_loss, val_loss, optimizer.param_groups[0]['lr'], elapsed_time))

        if args.early_stopping:
            early_stopping(val_loss, model, idx_epoch)
            if early_stopping.early_stop:
                print("early_stopping")
                break


    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Train Loss')
    plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='Validation Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title("Loss history")
    figure_path = "./figures/" + str(args.window_size) + "_loss_history.png"
    plt.savefig(figure_path)




if __name__ == '__main__':
    train()
