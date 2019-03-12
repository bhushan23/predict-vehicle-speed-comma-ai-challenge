import torch
import torch.nn as nn
import numpy as np
from data_loading import get_test_dataset
from torch.utils.data import DataLoader


def test(model, dl, write_to_file = None):
    prediction = None
    for _,img in enumerate(dl):
        if torch.cuda.is_available():
            img = img.cuda()

        temp = model(img)
        if prediction is None:
            prediction = temp
        else:
            prediction = torch.cat((prediction, temp))

    if write_to_file:
        prediction = prediction.view(-1)
        output = prediction.detach().cpu().numpy()
        np.savetxt(write_to_file, output, delimiter=',')

def predict(model, dl, write_to_file = None):
    prediction = None
    labels     = None
    mse_loss   = nn.MSELoss()
    for _, data in enumerate(dl):
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.float()
            label = label.cuda()
        temp = model(img)
        if prediction is None:
            prediction = temp
            labels     = label
        else:
            prediction = torch.cat((prediction, temp))
            labels     = torch.cat((labels, label))

    prediction = prediction.view(-1)
    if write_to_file:
        output = torch.stack((labels, prediction)).detach().cpu().numpy()
        np.savetxt(write_to_file, output, delimiter=',')

    return mse_loss(prediction, labels)


def train(model, train_dl, val_dl, test_data, num_epochs = 10, log_path = './metadata/'):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss = nn.MSELoss()
    test_dataset = get_test_dataset(test_data)
    test_dl = DataLoader(test_dataset, batch_size = 5)

    if torch.cuda.is_available():
        loss = loss.cuda()

    for epoch in range(1, num_epochs+1):
        tloss = 0
        for _, data in enumerate(val_dl):
            img, label = data
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            prediction = model(img)
            output     = loss(prediction, label.float())
            tloss      += output.item()
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        
        print('Epoch: {} Loss: {}'.format(epoch, tloss))
        if epoch % 1 == 0:
            print('Train MSE , Val MSE: {}'.format(predict(model, val_dl, write_to_file='Val_data.csv')))
            torch.save(model.cpu().state_dict(), log_path + 'model_'+str(epoch)+'.pkl')
            model = model.cuda()
            test(model, test_dl, write_to_file='test_output_'+str(epoch)+'.csv')





