import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda:'+'0') if torch.cuda.is_available() else torch.device('cpu')

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        pass

    def forward(self, feature, hidden):
        ratio = torch.bmm(hidden, feature)
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        ratio = F.softmax(ratio, dim=1).unsqueeze(2)
        result = torch.bmm(hidden.permute(0, 2, 1), ratio)
        result = result.view(result.size(0), -1)
        return result

class RNNAttention(nn.Module):
    def __init__(self, max_para_q,data_type):
        super(RNNAttention, self).__init__()
        self.input_dim = 768
        self.hidden_dim = 256
        self.dropout_rnn = 0
        self.dropout_fc = 0
        self.direction = 1
        self.num_layers = 1
        self.output_dim = 1

        self.max_para_q = max_para_q
        self.data_type=data_type

        self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True,
                           num_layers=self.num_layers, bidirectional=False, dropout=self.dropout_rnn)

        self.max_pool = nn.MaxPool1d(kernel_size=self.max_para_q)

        self.fc_a = nn.Linear(self.hidden_dim*self.direction, self.hidden_dim*self.direction)

        self.attention = Attention()
        self.fc_f = nn.Linear(self.hidden_dim*self.direction, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_fc)

    def forward(self, hidden_seq):
        batch_size = hidden_seq.size()[0]
        if self.data_type=='elam': # single gpu
            self.hidden = (
                torch.autograd.Variable(torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim)).to(device)),  # 1, b, h
                torch.autograd.Variable(torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim)).to(device))   # 1, b, h
            )
        else:
            self.hidden = (
                torch.autograd.Variable(
                    torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim)).cuda()),
                torch.autograd.Variable(
                    torch.zeros((self.direction * self.num_layers, batch_size, self.hidden_dim)).cuda())
            )

        rnn_out, self.hidden = self.rnn(hidden_seq, self.hidden)

        tmp_rnn = rnn_out.permute(0, 2, 1)

        feature = self.max_pool(tmp_rnn)
        feature = feature.squeeze(2)
        feature = self.fc_a(feature)
        feature = feature.unsqueeze(2)

        atten_out = self.attention(feature, rnn_out)
        atten_out = self.dropout(atten_out)

        y = self.fc_f(atten_out)

        y = torch.sigmoid(y)

        return y


