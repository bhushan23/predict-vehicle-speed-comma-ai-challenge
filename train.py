import torch
import torch.nn as nn
import numpy as np

def test(model, dl, write_to_file = None):
    prediction = None
    for _, data in enumerate(dl):
        img, _ = data
        if torch.cuda.is_available():
            img = img.cuda()

        temp = model(img)
        if prediction is None:
            prediction = temp
        else:
            prediction = torch.cat(prediction, temp, 0)

    if write_to_file:
        output = prediction.numpy()
        np.savetxt(write_to_file, output, delimiter=',')

    return nn.MSELoss(labels, prediction, reduction=True) * 100    


def predict(model, dl, write_to_file = None):
    prediction = None
    labels     = None
    for _, data in enumerate(dl):
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        temp = model(img)
        if prediction is None:
            prediction = temp
            labels     = label
        else:
            prediction = torch.cat((prediction, temp))
            labels     = torch.cat((labels, label))

    print('SHAPE: ', len(dl), prediction.shape, labels.shape)
    if write_to_file:
        output = torch.stack((labels, prediction)).numpy()
        np.savetxt(write_to_file, output, delimiter=',')

    return nn.MSELoss(labels, prediction, reduction=True) * 100    


def train(model, train_dl, val_dl, num_epochs = 10, log_path = './metadata/'):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss = nn.MSELoss()
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
            output     = loss(label.float(), prediction)
            tloss      += output.item()
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        
        print('Epoch: {} Loss: {}'.format(epoch, tloss))
        if epoch % 1 == 0:
            print('Train Accuracy: {}, Val Accuracy: {}'.format(predict(model, train_dl), predict(model, val_dl, write_to_file='Val_data.csv')))
            torch.save(model.cpu().state_dict(), log_path + 'model_'+str(epoch)+'.pkl')




