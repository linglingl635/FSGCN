import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class FSGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(FSGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1_1_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv1_1_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv1_3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv1_4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        
        self.conv2_1_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2_1_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2_3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2_4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.conv3_1_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3_1_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3_3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv3_4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.conv4_1_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv4_1_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv4_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv4_3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4_4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.conv5_1_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv5_1_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv5_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv5_3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv5_4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        #self.f = self.new_f
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    #def new_f(self, x):
    #    return torch.exp(- x ** 2)

    def forward(self, x, A=None, alpha = [1, 1]):
        #indices_upper = [x for x in range(12)] + [x for x in range(20, 25)]
        #indices_lower = [x for x in range(12, 20)]
        #x_upper = x[:, :, :, indices_upper]
        #x_lower = x[:, :, :, indices_lower]
        x_upper = x[:, :, :, 0:17]
        x_lower = x[:, :, :, 17:25]

        x_upper1, x_upper2 = self.conv1_1_1(x_upper).mean(-2), self.conv1_2(x_upper).mean(-2)
        x_a = self.tanh(x_upper1.unsqueeze(-1) - x_upper2.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv1_1_1(x_upper).mean(-2), self.conv1_2(x_lower).mean(-2)
        x_b = self.tanh(x_upper1.unsqueeze(-1) - x_lower1.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv1_2(x_upper).mean(-2), self.conv1_1_2(x_lower).mean(-2)
        x_c = self.tanh(x_lower1.unsqueeze(-1) - x_upper1.unsqueeze(-2))
        x_lower1, x_lower2 = self.conv1_1_2(x_lower).mean(-2), self.conv1_2(x_lower).mean(-2)
        x_d = self.tanh(x_lower1.unsqueeze(-1) - x_lower2.unsqueeze(-2))

        x_upper1_1 = torch.cat([x_a, x_b], dim=-1)
        x_lower1_2 = torch.cat([x_c, x_d], dim=-1)       
 
        x_upper1, x_upper2 = self.conv2_1_1(x_upper).mean(-2), self.conv2_2(x_upper).mean(-2)
        x_a = self.tanh(x_upper1.unsqueeze(-1) - x_upper2.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv2_1_1(x_upper).mean(-2), self.conv2_2(x_lower).mean(-2)
        x_b = self.tanh(x_upper1.unsqueeze(-1) - x_lower1.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv2_2(x_upper).mean(-2), self.conv2_1_2(x_lower).mean(-2)
        x_c = self.tanh(x_lower1.unsqueeze(-1) - x_upper1.unsqueeze(-2))
        x_lower1, x_lower2 = self.conv2_1_2(x_lower).mean(-2), self.conv2_2(x_lower).mean(-2)
        x_d = self.tanh(x_lower1.unsqueeze(-1) - x_lower2.unsqueeze(-2))
        
        #x1 = torch.cat([torch.cat([x_a, x_b], dim=-1), torch.cat([x_c, x_d], dim=-1)], dim=-2)
        x_upper2_1 = torch.cat([x_a, x_b], dim=-1)
        x_lower2_2 = torch.cat([x_c, x_d], dim=-1)

        x_upper1, x_upper2 = self.conv3_1_1(x_upper).mean(-2), self.conv3_2(x_upper).mean(-2)
        x_a = self.tanh(x_upper1.unsqueeze(-1) - x_upper2.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv3_1_1(x_upper).mean(-2), self.conv3_2(x_lower).mean(-2)
        x_b = self.tanh(x_upper1.unsqueeze(-1) - x_lower1.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv3_2(x_upper).mean(-2), self.conv3_1_2(x_lower).mean(-2)
        x_c = self.tanh(x_lower1.unsqueeze(-1) - x_upper1.unsqueeze(-2))
        x_lower1, x_lower2 = self.conv3_1_2(x_lower).mean(-2), self.conv3_2(x_lower).mean(-2)
        x_d = self.tanh(x_lower1.unsqueeze(-1) - x_lower2.unsqueeze(-2))

        x_upper3_1 = torch.cat([x_a, x_b], dim=-1)
        x_lower3_2 = torch.cat([x_c, x_d], dim=-1)

        x_upper1, x_upper2 = self.conv4_1_1(x_upper).mean(-2), self.conv4_2(x_upper).mean(-2)
        x_a = self.tanh(x_upper1.unsqueeze(-1) - x_upper2.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv4_1_1(x_upper).mean(-2), self.conv4_2(x_lower).mean(-2)
        x_b = self.tanh(x_upper1.unsqueeze(-1) - x_lower1.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv4_2(x_upper).mean(-2), self.conv4_1_2(x_lower).mean(-2)
        x_c = self.tanh(x_lower1.unsqueeze(-1) - x_upper1.unsqueeze(-2))
        x_lower1, x_lower2 = self.conv4_1_2(x_lower).mean(-2), self.conv4_2(x_lower).mean(-2)
        x_d = self.tanh(x_lower1.unsqueeze(-1) - x_lower2.unsqueeze(-2))

        x_upper4_1 = torch.cat([x_a, x_b], dim=-1)
        x_lower4_2 = torch.cat([x_c, x_d], dim=-1)

        x_upper1, x_upper2 = self.conv5_1_1(x_upper).mean(-2), self.conv5_2(x_upper).mean(-2)
        x_a = self.tanh(x_upper1.unsqueeze(-1) - x_upper2.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv5_1_1(x_upper).mean(-2), self.conv5_2(x_lower).mean(-2)
        x_b = self.tanh(x_upper1.unsqueeze(-1) - x_lower1.unsqueeze(-2))
        x_upper1, x_lower1 = self.conv5_2(x_upper).mean(-2), self.conv5_1_2(x_lower).mean(-2)
        x_c = self.tanh(x_lower1.unsqueeze(-1) - x_upper1.unsqueeze(-2))
        x_lower1, x_lower2 = self.conv5_1_2(x_lower).mean(-2), self.conv5_2(x_lower).mean(-2)
        x_d = self.tanh(x_lower1.unsqueeze(-1) - x_lower2.unsqueeze(-2))

        x_upper5_1 = torch.cat([x_a, x_b], dim=-1)
        x_lower5_2 = torch.cat([x_c, x_d], dim=-1)


        x_upper1_1, x_upper2_1, x_upper3_1, x_upper4_1, x_upper5_1 = x_upper2_1, x_upper3_1, x_upper4_1, x_upper5_1, x_upper1_1
        #print(x_upper1.shape, x_lower1.shape)
        x1 = torch.cat([x_upper1_1, x_lower1_2], dim=-2)
        x2 = torch.cat([x_upper2_1, x_lower2_2], dim=-2)
        x3 = torch.cat([x_upper3_1, x_lower3_2], dim=-2)
        x4 = torch.cat([x_upper4_1, x_lower4_2], dim=-2)
        x5 = torch.cat([x_upper5_1, x_lower5_2], dim=-2)

        # N, C, V, V
        graph1 = self.conv1_4(x1)
        x1_1 = graph1 * alpha[0] + (A[0].unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1_2 = self.conv1_3(x)
        x1 = torch.einsum('ncuv,nctv->nctu', x1_1, x1_2)

        graph2 = self.conv2_4(x2)
        x2_1 = graph2 * alpha[1] + (A[1].unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x2_2 = self.conv2_3(x)
        x2 = torch.einsum('ncuv,nctv->nctu', x2_1, x2_2)

        graph3 = self.conv3_4(x3)
        x3_1 = graph3 * alpha[2] + (A[2].unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x3_2 = self.conv3_3(x)
        x3 = torch.einsum('ncuv,nctv->nctu', x3_1, x3_2)

        graph4 = self.conv4_4(x4)
        x4_1 = graph4 * alpha[3] + (A[3].unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x4_2 = self.conv4_3(x)
        x4 = torch.einsum('ncuv,nctv->nctu', x4_1, x4_2)

        graph5 = self.conv5_4(x5)
        x5_1 = graph5 * alpha[4] + (A[4].unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x5_2 = self.conv5_3(x)
        x5 = torch.einsum('ncuv,nctv->nctu', x5_1, x5_2)

        graph_list = []
        graph_list.append(graph1)
        graph_list.append(graph2)
        graph_list.append(graph3)
        graph_list.append(graph4)
        graph_list.append(graph5)
        return x1 + x2 + x3 + x4 + x5,  graph_list

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = FSGC(in_channels, out_channels)
        #self.convs = nn.ModuleList()
        #for i in range(self.num_subset):
        #   self.convs.append(FSGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(self.num_subset))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        graph_list = []
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        #for i in range(self.num_subset):
        #    z, graph = self.convs[i](x, A[i], self.alpha)
        #   graph_list.append(graph)
        #    y = z + y if y is not None else z
        y, graph_list = self.convs(x, A, self.alpha)
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        
        return y, torch.stack(graph_list, 1)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        # self.tcn1 = TemporalConv(out_channels, out_channels, stride=stride)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        z, graph = self.gcn1(x)
        y = self.relu(self.tcn1(z) + self.residual(x))
        return y, graph


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A_c = self.graph.A # 3,25,25
        mean_A = np.mean(A_c, axis=0, keepdims=True)
        A_c = np.concatenate([A_c, mean_A, mean_A], axis=0)
        A = A_c.copy()
        
        A[:, 12:17, :], A[:, 17:25, :] = A_c[:, 20:25, :], A_c[:, 12:20, :]
        A[:, :, 12:17], A[:, :, 17:25] = A_c[:, :, 20:25], A_c[:, :, 12:20]

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        #print(x.size())
        #exit()
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        #print(x.size())
        #exit()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x_c = x.clone()
        x_c[:, :, :, 12:17], x_c[:, :, :, 17:25] = x[:, :, :, 20:25], x[:, :, :, 12:20]
        x = x_c

        x, _ = self.l1(x)
        x, _ = self.l2(x)
        x, _ = self.l3(x)
        x, _ = self.l4(x)
        x, _ = self.l5(x)
        x, _ = self.l6(x)
        x, _ = self.l7(x)
        x, _ = self.l8(x)
        x, _ = self.l9(x)
        x, graph = self.l10(x)
        
        graph_upper, graph_lower = graph[:, :, :, :, 0:17], graph[:, :, :, :, 17:25]
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        graph1 = graph_upper.view(N, M, -1, c_new, V, 17)
        # graph4 = torch.einsum('n m k c u v, n m k c v l -> n m k c u l', graph2, graph2)
        graph1 = graph1.view(N, M, -1, c_new, V, 17).mean(1).mean(2).view(N, -1)
        # graph4 = graph4.view(N, M, -1, c_new, V, V).mean(1).mean(2).view(N, -1)
        # graph = torch.cat([graph2, graph4], -1)
        graph2 = graph_lower.view(N, M, -1, c_new, V, 8)
        graph2 = graph2.view(N, M, -1, c_new, V, 8).mean(1).mean(2).view(N, -1)
        return self.fc(x), graph1, graph2

