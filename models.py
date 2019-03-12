import torch
import torch.nn as nn

def conv(batchNorm, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(dropout),
        nn.Dropout(dropout)
    )

class DeepVO(nn.Module):
    def __init__(self, image_w, image_h, seq_len=2, batchNorm=True):
        super(DeepVO, self).__init__()

        # CNN: 
        self.batchNorm = batchNorm
        self.conv1     = conv(self.batchNorm,   6,   64, kernel_size=7, stride=1)
        self.conv2     = conv(self.batchNorm,  64,  128, kernel_size=5, stride=1)
        self.conv3     = conv(self.batchNorm, 128,  256, kernel_size=5, stride=1)
        self.conv3_1   = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1)
        self.conv4     = conv(self.batchNorm, 256,  512, kernel_size=3, stride=1)
        self.conv4_1   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1)
        self.conv5     = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1)
        self.conv5_1   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1)
        self.conv6     = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=1)

        __test_dim = torch.zeros(1, 6, image_w, image_h)
        __test_dim = self.extract_features(__test_dim)
        self.hidden_size  = 3
        self.seq_len      = seq_len

        self.rnn   = nn.LSTM(
                        input_size  = int(__test_dim.numel()),
                        hidden_size = self.hidden_size,
                        num_layers   = 2,
                        dropout     = 0,
                        batch_first = True  
                    )
        self.rnn_drop_out = nn.Dropout(0.0)
        self.linear       = nn.Linear(in_features=self.hidden_size, out_features=1)
        print('Model initialized correctly!')

    def forward(self, x):
        # x          = torch.cat((x[:,:-1], x[:,1:]), dim=2)
        batch_size = x.shape[0]
        seq_len    = self.seq_len

        # CNN 
        print(x.shape)
        x = self.extract_features(x)
        x = x.view(batch_size, seq_len, -1)

        # RNN
        out, hc = self.rnn(x)
        out     = self.rnn_drop_out(out)
        out     = self.linear(out)
        return out


    def extract_features(self, x):
        print(x.shape)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3_1(self.conv3(y))
        y = self.conv4_1(self.conv4(y))
        y = self.conv5_1(self.conv5(y))
        y = self.conv6(y)
        return y
        
        
class DeepVO_small(nn.Module):
    def __init__(self, image_w, image_h, seq_len=2, batchNorm=True):
        super(DeepVO_small, self).__init__()

        # CNN: 
        self.batchNorm = batchNorm
        self.conv1     = conv(self.batchNorm,   6,   64, kernel_size=5, stride=1, padding = 2)
        self.conv2     = conv(self.batchNorm,  64,  128, kernel_size=5, stride=1, padding = 2)

        __test_dim = torch.zeros(1, 6, image_w, image_h)
        __test_dim = self.extract_features(__test_dim)
        self.hidden_size  = 3
        self.seq_len      = seq_len
        print(int(__test_dim.numel()))
        self.rnn   = nn.LSTM(
                        input_size  = int(__test_dim.numel() / seq_len),
                        hidden_size = self.hidden_size,
                        num_layers   = 2,
                        dropout     = 0,
                        batch_first = True  
                    )
        self.rnn_drop_out = nn.Dropout(0.0)
        self.linear       = nn.Linear(in_features=6, out_features=1)
        print('Model initialized correctly!')

    def forward(self, x):
        # x          = torch.cat((x[:,:-1], x[:,1:]), dim=2)
        batch_size = x.shape[0]
        seq_len    = self.seq_len

        # CNN 
        # print(x.shape)
        x = self.extract_features(x)
        # print('After CNN: ', x.shape)
        x = x.view(batch_size, seq_len, -1)
        # print('Flatten: ', x.shape)
        # RNN
        out, hc = self.rnn(x)
        out     = self.rnn_drop_out(out)
        out     = out.contiguous().view(batch_size, -1)
        out     = self.linear(out)
        return out


    def extract_features(self, x):
        # print(x.shape)
        y = self.conv1(x)
        y = self.conv2(y)
        return y
        
        