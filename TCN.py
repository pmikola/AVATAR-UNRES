from torch import nn
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, features, output_size, layers=2, kernel_s=3, filters=100, dropout=0.2):
        super(TCN, self).__init__()
        self.output_size = output_size
        self.features = features
        self.layers = layers
        self.kernel_s = kernel_s
        self.filters = filters
        self.dropout = dropout

        self.conv1 = nn.Conv1d(1, self.filters, kernel_size=(kernel_s,), dilation=1,
                               padding=(kernel_s - 1) // 2, stride=(1,))
        # self.conv1 = nn.Conv1d(self.features, self.filters, kernel_size=(kernel_s, kernel_s), dilation=(1, 1),
        #                        padding=(kernel_s - 1) // 2, stride=(1,))
        self.bn1 = nn.BatchNorm1d(self.filters)
        self.dropout1 = nn.Dropout(self.dropout)

        self.conv2 = nn.Conv1d(self.filters, self.features, kernel_size=(kernel_s,), dilation=1,
                               padding=(kernel_s - 1) // 2, stride=(1,))
        self.bn2 = nn.BatchNorm1d(self.features)
        self.dropout2 = nn.Dropout(self.dropout)

        self.conv_1x1 = nn.Conv1d(1, self.features, kernel_size=(1,), )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()

        for i in range(layers - 1):
            dilation_factor = 2 ** (i + 1)
            self.convs.append(
                nn.Conv1d(self.features, self.filters, kernel_size=(kernel_s,), dilation=dilation_factor,
                          padding=(kernel_s - 1) * dilation_factor // 2, stride=(1,)))
            self.bns.append(nn.BatchNorm1d(self.filters))
            self.convs2.append(
                nn.Conv1d(self.filters, self.features, kernel_size=(kernel_s,), dilation=dilation_factor,
                          padding=(kernel_s - 1) * dilation_factor // 2, stride=(1,)))
            self.bns2.append(nn.BatchNorm1d(self.features))
            self.convs_1x1.append(
                nn.Conv1d(self.features, self.features, kernel_size=(1,)))

        self.out = nn.Linear(self.features, self.output_size,bias=False)

    def forward(self, x):
        # x = torch.swapaxes(x, 2, 1)
        # x = x.permute(0, 2, 1)
        res = self.conv_1x1(x)  # optional?
        # print(x.shape, 'x_res')

        x = self.conv1(x)
        # print(x.shape,'x1')
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        o = F.gelu(x)
        # print(o.shape,'o0')
        x = o + res

        for i in range(self.layers - 1):
            res = self.convs_1x1[i](x)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.gelu(x)
            x = self.convs2[i](x)
            x = self.bns2[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            o = F.gelu(x)
            # print(x.shape, i)
            x = o + res
            x = F.gelu(x)

        # x = x.permute(0, 2, 1)
        x = x[:, -1, :]
        # print(x.shape)
        x = F.gelu(self.out(x))
        return x