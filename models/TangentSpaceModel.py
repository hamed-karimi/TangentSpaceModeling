import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_output_vectors, enable_bn):
        super(Model, self).__init__()
        self.n_output_vectors = n_output_vectors

        self.linear1 = LinearBlock2D(input_dim=9*128, hidden_dim=256, output_dim=64, layers=2, enable_bn=enable_bn)
        self.linear2 = LinearBlock2D(input_dim=64, hidden_dim=64, output_dim=64, layers=4, enable_bn=enable_bn)
        self.linear3 = LinearBlock2D(input_dim=64, hidden_dim=128, output_dim=2*128, layers=2, enable_bn=enable_bn)
        self.linear4 = LinearBlock2D(input_dim=2*128, hidden_dim=3*128, output_dim=4*128, layers=2, enable_bn=enable_bn)
        self.linear5 = LinearBlock2D(input_dim=4*128, hidden_dim=6*128, output_dim=8*128, layers=2, enable_bn=enable_bn)
        self.linear6 = LinearBlock2D(input_dim=8*128, hidden_dim=12*128, output_dim=16*128, layers=2, enable_bn=enable_bn)
        self.linear7 = LinearBlock2D(input_dim=16*128, hidden_dim=24*128, output_dim=32*128, layers=2, enable_bn=enable_bn)
        self.linear8 = LinearBlock2D(input_dim=32*128, hidden_dim=48*128, output_dim=6*9*128, layers=2, enable_bn=enable_bn)
        self.out = nn.Linear(in_features=6*9*128, out_features=6*9*128)

        # self.conv_block1 = ConvBlock2D(input_dim=128, output_dim=256, hidden_dim=128, layers=2, enable_bn=enable_bn)
        # self.conv_block2 = ConvBlock2D(input_dim=256, output_dim=512, hidden_dim=256, layers=2, enable_bn=enable_bn)
        # self.conv_block3 = ConvBlock2D(input_dim=512, output_dim=768, hidden_dim=512, layers=2, enable_bn=enable_bn)
        # self.upsample_3d_1 = nn.ConvTranspose3d(in_channels=768, out_channels=512, kernel_size=(1, 1, 2), stride=1)
        # self.conv_block4 = ConvBlock3D(input_dim=512, output_dim=512, hidden_dim=512, layers=2, enable_bn=enable_bn)
        # self.upsample_3d_2 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=(1, 1, 3), stride=1)
        # self.conv_block5 = ConvBlock3D(input_dim=256, output_dim=256, hidden_dim=256, layers=2, enable_bn=enable_bn)
        # self.upsample_3d_3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(1, 1, 3), stride=1)
        # self.conv_block6 = ConvBlock3D(input_dim=128, output_dim=128, hidden_dim=128, layers=2, enable_bn=enable_bn)
        # # self.linear_out = nn.Linear(in_features=128, out_features=self.n_output_vectors)
        # self.conv_out = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1, groups=1)

    def forward(self, z1):
        x = torch.view_copy(z1, (z1.shape[0], -1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        x = self.linear8(x)
        x = self.out(x)

        return x.view(x.shape[0], -1, 6)

class LinearBlock2D(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(LinearBlock2D, self).__init__()
        if layers == 1:
            layer = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.add_module('0 LinearBlock2D', layer)
        else:
            for i in range(layers):
                if i == 0:
                    layer = nn.Sequential(
                        nn.Linear(in_features=input_dim, out_features=hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                    )

                elif i == (layers - 1):
                    layer = nn.Sequential(
                        nn.Linear(in_features=hidden_dim, out_features=output_dim),
                        nn.BatchNorm1d(output_dim),
                        nn.ReLU(inplace=True),
                    )
                else:
                    layer = nn.Sequential(
                        nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                    )

                self.add_module('%d ConvBlock2D' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


# class ConvLayer2D(nn.Module):
#
#     def __init__(self, input_dim, output_dim, enable_bn):
#         super(ConvLayer2D, self).__init__()
#
#         if enable_bn:
#             self.layer = nn.Sequential(
#                 nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
#                 # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
#                 nn.BatchNorm2d(output_dim),
#                 nn.ReLU(inplace=True),
#             )
#         else:
#             self.layer = nn.Sequential(
#                 nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
#                 # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
#                 nn.ReLU(inplace=True),
#             )
#
#     def forward(self, x):
#
#         return self.layer(x)
#
#
# class ConvBlock3D(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):
#
#         super(ConvBlock3D, self).__init__()
#
#         if layers == 1:
#
#             layer = ConvLayer3D(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)
#
#             self.add_module('0 ConvBlock3D', layer)
#
#         else:
#
#             for i in range(layers):
#
#                 if i == 0:
#                     layer = ConvLayer3D(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
#                 elif i == (layers - 1):
#                     layer = ConvLayer3D(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
#                 else:
#                     layer = ConvLayer3D(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
#
#                 self.add_module('%d ConvBlock3D' % i, layer)
#
#         # maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # self.add_module('%d MaxPooling' % layers, maxpool)
#
#     def forward(self, x):
#
#         for name, layer in self.named_children():
#             x = layer(x)
#
#         return x
#
#
# class ConvLayer3D(nn.Module):
#
#     def __init__(self, input_dim, output_dim, enable_bn):
#         super(ConvLayer3D, self).__init__()
#
#         if enable_bn:
#             self.layer = nn.Sequential(
#                 nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=(3, 3, 3), stride=1, padding=1, groups=1),
#                 # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
#                 nn.BatchNorm3d(output_dim),
#                 nn.ReLU(inplace=True),
#             )
#         else:
#             self.layer = nn.Sequential(
#                 nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=(3, 3, 3), stride=1, padding=1, groups=1),
#                 # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
#                 nn.ReLU(inplace=True),
#             )
#
#     def forward(self, x):
#
#         return self.layer(x)

