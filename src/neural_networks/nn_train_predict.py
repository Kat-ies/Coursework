"""
functions:

nn_training(x_train, y_train, is_gpu, loop,
time_df, feature_type, index,y_points_acc, y_points_loss,
y_points_val, net_type, learning_rate)

nn_test(x_test, is_gpu, feature_type,
data_df, time_df, index, net_type)
"""

from constants import *
from visualization import plot_graphic
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import numpy as np
from data_saver import save_nn_model
from data_loader import load_nn_model
from sklearn.metrics import accuracy_score
from nn_models import *


def nn_training(x_train, y_train, is_gpu, time_df, feature_type, index,
                y_points_acc, y_points_loss, y_points_val, net_type,
                loop=MAX_ITER, learning_rate=LEARNING_RATE):
    """
    function nn_training(x_train, y_train, is_gpu, time_df,
    feature_type, index, y_points_acc, y_points_loss,
    y_points_val, net_type, loop=MAX_ITER, learning_rate=LEARNING_RATE)
    train the model. Here we can set feature type, the epoch number,
    learning rate and processesor type.

    For a stop criteria is used early stopping: if during 100
    epoch we don't have any accuracy changes on
    validate data, the training stops.

    Note: for realization this point 25% from test
    samples will be used as validation samples.

    To analyse the whole training process there are some graphics:

    • accuracy on test samples
    • accuracy on validation samples
    • loss function values

    The time for training in cpu and gpu will be also measured.
    """
    # Let's fix everything that can be fixed to get the same results between runs
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    if net_type == 'convolution':
        net = ConvNet()
        net_index = 1
    else:
        net = Net(len(x_train[0]))
        net_index = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)

    # arrays of points for plotting accuracy graphics
    y_points_for_one, y_loss_points_for_one, y_val_points_for_one = [], [], []

    t0 = time.time()

    # separate x_train, to get  x_validate
    x_train, x_validate = np.split(x_train, [int(0.75 * len(x_train))])
    y_train, y_validate = np.split(y_train, [int(0.75 * len(y_train))])

    x_train_torch, x_validate_torch = torch.from_numpy(x_train).float(), \
                                      torch.from_numpy(x_validate).float()
    y_train_torch, y_validate_torch = torch.from_numpy(y_train), torch.from_numpy(y_validate)

    if net_type == 'convolution':
        x_train_torch = torch.unsqueeze(torch.reshape(x_train_torch, \
                                                      (len(x_train_torch), 25, 25)), 1)
        x_validate_torch = torch.unsqueeze(torch.reshape(x_validate_torch, \
                                                         (len(x_validate_torch), 25, 25)), 1)

    # for using gpu and cpu
    if is_gpu == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net.to(device)

    x_train_torch, y_train_torch = x_train_torch.to(device), y_train_torch.to(device)
    x_validate_torch, y_validate_torch = x_validate_torch.to(device), \
                                         y_validate_torch.to(device)
    print('\n', net_type, ' network')
    print('Feature type: ', feature_type)
    print(device)

    for epoch in range(loop):  # loop
        optimizer.zero_grad()
        output = net(x_train_torch)
        loss = criterion(output, y_train_torch)
        loss.backward()
        # clip_grad_norm(net.parameters(), 0.01)
        optimizer.step()

        with torch.no_grad():
            validate_output = net(x_validate_torch)

        y_loss_points_for_one.append(loss.item())
        y_points_for_one.append\
            (accuracy_score(y_train, torch.argmax(output.data, dim=1).cpu()))
        y_val_points_for_one.append\
            (accuracy_score(y_validate, torch.argmax(validate_output.data, dim=1).cpu()))

        if len(y_val_points_for_one) >= 100 and \
                np.all(y_val_points_for_one[-100:] == y_val_points_for_one[-1]):
            print('Early stopping! No changes in last '
                  '100 epoch for validation samples! Epoch: ', epoch)
            break
    t = time.time()
    print('Last validation accuracy: ', y_val_points_for_one[-1])
    print('Last train accuracy: ', y_points_for_one[-1])
    print('Loss: ', y_loss_points_for_one[-1])

    # save trained model
    save_nn_model(net.state_dict(),  feature_type + '_' + net_type + '_neural_net')

    # fill the table with time
    time_df.loc[CATEGORIES[index]][NET_COLS[is_gpu + 2 * net_index]] += t - t0

    # что-то не чистится:(
    del x_train_torch, y_train_torch, x_validate_torch, y_validate_torch
    torch.cuda.empty_cache()

    # plot graphic
    if is_gpu == 1:
        y_points_acc.append(y_points_for_one)
        y_points_loss.append(y_loss_points_for_one)
        y_points_val.append(y_val_points_for_one)
        if len(y_points_acc) == 3 or net_type == 'convolution':
            plot_graphic(y_points_val, 'Accuracy score for validation')
            plot_graphic(y_points_acc, 'Accuracy score for train')
            plot_graphic(y_points_loss, 'Loss')


def nn_test(x_test, is_gpu, feature_type, data_df, time_df, index, net_type):
    """
    function nn_test(x_test, is_gpu, feature_type,
    data_df, time_df, index, net_type) tests
    the trained models and measure time that
    is need to classify one object using trained models.
    """
    x_test_torch = torch.from_numpy(x_test).float()

    if net_type == 'convolution':
        net = ConvNet()
        net_index = 1
        x_test_torch = torch.unsqueeze\
            (torch.reshape(x_test_torch, (len(x_test_torch), 25, 25)), 1)
    else:
        net = Net(len(x_test[0]))
        net_index = 0
    net.load_state_dict(load_nn_model(feature_type + '_' + net_type + '_neural_net'))

    t0 = time.time()

    with torch.no_grad():
        output = net(x_test_torch)
        data_df[NET_COLS[is_gpu + 2 * net_index]] = torch.argmax(output.data, dim=1)
    t = time.time()

    del x_test_torch
    torch.cuda.empty_cache()

    time_df.loc[CATEGORIES[index]][NET_COLS[is_gpu + 2 * net_index]] += (t - t0) / len(x_test)
