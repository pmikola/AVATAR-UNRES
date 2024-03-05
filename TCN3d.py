from torch import nn
import torch.nn.functional as F


class TCN3d(nn.Module):
    def __init__(self, features, output_size, layers=1, kernel_s=3, filters=100, dropout=0.2):
        super(TCN3d, self).__init__()
        self.output_size = output_size
        self.features = features
        self.layers = layers
        self.kernel_s = kernel_s
        self.filters = filters
        self.dropout = dropout

        self.conv1 = nn.Conv3d(1, self.filters, kernel_size=(kernel_s, kernel_s, kernel_s), dilation=1,
                               padding=(kernel_s - 1) // 2, stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(self.filters)
        self.dropout1 = nn.Dropout3d(self.dropout)

        self.conv2 = nn.Conv3d(self.filters, self.features, kernel_size=(kernel_s, kernel_s, kernel_s), dilation=1,
                               padding=(kernel_s - 1) // 2, stride=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(self.features)
        self.dropout2 = nn.Dropout3d(self.dropout)

        self.conv_1x1 = nn.Conv3d(1, self.features, kernel_size=(1, 1, 1))

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
                nn.Conv3d(self.features, self.filters, kernel_size=(kernel_s, kernel_s, kernel_s),
                          dilation=dilation_factor,
                          padding=(kernel_s - 1) * dilation_factor // 2, stride=(1, 1, 1)))
            self.bns.append(nn.BatchNorm3d(self.filters))
            self.convs2.append(
                nn.Conv3d(self.filters, self.features, kernel_size=(kernel_s, kernel_s, kernel_s),
                          dilation=dilation_factor,
                          padding=(kernel_s - 1) * dilation_factor // 2, stride=(1, 1, 1)))
            self.bns2.append(nn.BatchNorm3d(self.features))
            self.convs_1x1.append(
                nn.Conv3d(self.features, self.features, kernel_size=(1, 1, 1)))

        self.out = nn.Linear(self.features, self.output_size, bias=False)

    def forward(self, x):
        res = self.conv_1x1(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        o = F.gelu(x)

        x = o + res

        for i in range(self.layers - 1):
            print(x.shape)
            res = self.convs_1x1[i](x)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.dropouts[i](x)
            x = F.gelu(x)
            x = self.convs2[i](x)
            x = self.bns2[i](x)
            x = self.dropouts2[i](x)
            o = F.gelu(x)

            x = o + res
            x = F.gelu(x)

        x = x[:, -1, :, :, :]
        x = x.view(x.size(0), -1)
        x = F.gelu(self.out(x))
        return x
