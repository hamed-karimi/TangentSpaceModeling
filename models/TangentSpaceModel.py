import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_output_vectors, enable_bn):
        super(Model, self).__init__()
        self.n_output_vectors = n_output_vectors

        # Encoder (Downsampling path)
        self.encoder1 = LinearBlock2D(input_dim=9 * 128, hidden_dim=8 * 128, output_dim=7 * 128, layers=2, enable_bn=enable_bn)
        self.encoder2 = LinearBlock2D(input_dim=7 * 128, hidden_dim=6 * 128, output_dim=5 * 128, layers=2, enable_bn=enable_bn)
        self.encoder3 = LinearBlock2D(input_dim=5 * 128, hidden_dim=3 * 128, output_dim=1 * 128, layers=2, enable_bn=enable_bn)
        self.encoder4 = LinearBlock2D(input_dim=1 * 128, hidden_dim=64, output_dim=None, layers=1, enable_bn=enable_bn)

        # Bottleneck
        self.bottleneck = LinearBlock2D(input_dim=64, hidden_dim=64, output_dim=None, layers=2, enable_bn=enable_bn)

        # Decoder (Upsampling path) with skip connections
        self.decoder4 = LinearBlock2D(input_dim=64 + 64, hidden_dim=2 * 128, output_dim=1 * 128, layers=2, enable_bn=enable_bn)  # 64 + skip from encoder4
        self.decoder3 = LinearBlock2D(input_dim=1 * 128 + 1 * 128, hidden_dim=3 * 128, output_dim=5 * 128, layers=2, enable_bn=enable_bn)  # 1*128 + skip from encoder3
        self.decoder2 = LinearBlock2D(input_dim=5 * 128 + 5 * 128, hidden_dim=6 * 128, output_dim=7 * 128, layers=2, enable_bn=enable_bn)  # 5*128 + skip from encoder2
        self.decoder1 = LinearBlock2D(input_dim=7 * 128 + 7 * 128, hidden_dim=8 * 128, output_dim=9 * 128, layers=2, enable_bn=enable_bn)  # 7*128 + skip from encoder1

        # Additional dense layers for refinement
        self.dense1 = LinearBlock2D(input_dim=9 * 128, hidden_dim=12 * 128, output_dim=16 * 128, layers=2, enable_bn=enable_bn)
        self.dense2 = LinearBlock2D(input_dim=16 * 128, hidden_dim=24 * 128, output_dim=3 * 9 * 128, layers=2, enable_bn=enable_bn)
        self.dense3 = LinearBlock2D(input_dim=3 * 9 * 128, hidden_dim=3 * 9 * 128, output_dim=3 * 9 * 128, layers=4, enable_bn=enable_bn)

        # Output layer
        self.out = nn.Linear(in_features=3 * 9 * 128, out_features=3 * 9 * 128)

    def forward(self, z1):
        x = torch.view_copy(z1, (z1.shape[0], -1))

        # Encoder path (store skip connections)
        enc1 = self.encoder1(x)  # 7 * 128
        enc2 = self.encoder2(enc1)  # 5 * 128
        enc3 = self.encoder3(enc2)  # 1 * 128
        enc4 = self.encoder4(enc3)  # 64

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # 64

        # Decoder path (with skip connections)
        dec4 = self.decoder4(torch.cat([bottleneck, enc4], dim=1))  # Concatenate with enc4 skip
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))  # Concatenate with enc3 skip
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))  # Concatenate with enc2 skip
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))  # Concatenate with enc1 skip

        # Dense refinement layers
        x = self.dense1(dec1)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.out(x)

        return x.view(x.shape[0], -1, self.n_output_vectors)

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

                self.add_module('%d LinearBlock2D' % i, layer)

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

