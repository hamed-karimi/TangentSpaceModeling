import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_output_vectors, enable_bn):
        super(Model, self).__init__()
        self.n_output_vectors = n_output_vectors

        self.conv_block1 = ConvBlock2D(input_dim=128, output_dim=256, hidden_dim=128, layers=2, enable_bn=enable_bn)
        self.conv_block2 = ConvBlock2D(input_dim=256, output_dim=512, hidden_dim=256, layers=2, enable_bn=enable_bn)
        self.conv_block3 = ConvBlock2D(input_dim=512, output_dim=768, hidden_dim=512, layers=2, enable_bn=enable_bn)
        self.upsample_3d_1 = nn.ConvTranspose3d(in_channels=768, out_channels=512, kernel_size=(1, 1, 2), stride=1)
        self.conv_block4 = ConvBlock3D(input_dim=512, output_dim=512, hidden_dim=512, layers=2, enable_bn=enable_bn)
        self.upsample_3d_2 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=(1, 1, 3), stride=1)
        self.conv_block5 = ConvBlock3D(input_dim=256, output_dim=256, hidden_dim=256, layers=2, enable_bn=enable_bn)
        self.upsample_3d_3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(1, 1, 3), stride=1)
        self.conv_block6 = ConvBlock3D(input_dim=128, output_dim=128, hidden_dim=128, layers=2, enable_bn=enable_bn)
        # self.linear_out = nn.Linear(in_features=128, out_features=self.n_output_vectors)
        self.conv_out = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1, groups=1)

    def forward(self, z1):
        x = self.conv_block1(z1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.unsqueeze(-1)
        x = self.upsample_3d_1(x)
        x = self.conv_block4(x)
        x = self.upsample_3d_2(x)
        x = self.conv_block5(x)
        x = self.upsample_3d_3(x)
        x = self.conv_block6(x)
        x = self.conv_out(x)

        return x

class ConvBlock2D(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(ConvBlock2D, self).__init__()

        if layers == 1:

            layer = ConvLayer2D(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 ConvBlock2D', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = ConvLayer2D(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = ConvLayer2D(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = ConvLayer2D(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d ConvBlock2D' % i, layer)

        # maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class ConvLayer2D(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(ConvLayer2D, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)


class ConvBlock3D(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(ConvBlock3D, self).__init__()

        if layers == 1:

            layer = ConvLayer3D(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 ConvBlock3D', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = ConvLayer3D(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = ConvLayer3D(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = ConvLayer3D(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d ConvBlock3D' % i, layer)

        # maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class ConvLayer3D(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(ConvLayer3D, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=(3, 3, 3), stride=1, padding=1, groups=1),
                # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm3d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=(3, 3, 3), stride=1, padding=1, groups=1),
                # nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)

