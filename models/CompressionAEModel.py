import torch
import torch.nn as nn
from collections import OrderedDict


def get_configs(arch='vgg16'):
    if arch == 'vgg11':
        configs = [1, 1, 2, 2, 2]
    elif arch == 'vgg13':
        configs = [2, 2, 2, 2, 2]
    elif arch == 'vgg16':
        configs = [2, 2, 3, 3, 3]
    elif arch == 'vgg19':
        configs = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Undefined model")

    return configs


class EncodingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = VGGEncoder(configs=get_configs('vgg11'))
        self.encoder_transition = EncoderTransitionBlock(enable_bn=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_transition(x)

        return x

def load_encoding_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoding_model = EncodingModel().to(device)
    snapshot = torch.load('./weights/encoding model/snapshot_20.pth', map_location=device)
    model_dict = encoding_model.state_dict()
    new_state_dict = OrderedDict() #deepcopy(snapshot['state_dict'])
    for key in model_dict.keys():
        if f'encoder' in key:
            if f'module.{key}' in snapshot['state_dict'].keys():
                new_state_dict[key] = snapshot['state_dict'][f'module.{key}']

    print('encoding model:', encoding_model.load_state_dict(new_state_dict, strict=False))
    encoding_model.eval()

    return encoding_model

class DecodingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_transition = DecoderTransitionBlock(enable_bn=True)
        self.decoder = VGGDecoder(configs=get_configs('vgg11')[::-1])

    def forward(self, x):
        x = self.decoder_transition(x)
        x = self.decoder(x)
        return x

def load_decoding_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    decoding_model = DecodingModel().to(device)
    snapshot = torch.load('./weights/encoding model/snapshot_20.pth', map_location=device)
    model_dict = decoding_model.state_dict()
    new_state_dict = OrderedDict()  # deepcopy(snapshot['state_dict'])
    for key in model_dict.keys():
        if f'decoder' in key:
            if f'module.{key}' in snapshot['state_dict'].keys():
                new_state_dict[key] = snapshot['state_dict'][f'module.{key}']

    print('decoding model:', decoding_model.load_state_dict(new_state_dict, strict=False))
    decoding_model.eval()

    return decoding_model


class VGGAutoEncoder(nn.Module):

    def __init__(self, configs):
        super(VGGAutoEncoder, self).__init__()

        # VGG without Bn as AutoEncoder is hard to train
        self.encoder = VGGEncoder(configs=configs, enable_bn=True)
        self.encoder_transition = EncoderTransitionBlock(enable_bn=True)
        self.decoder_transition = DecoderTransitionBlock(enable_bn=True)
        self.decoder = VGGDecoder(configs=configs[::-1], enable_bn=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_transition(x)
        x = self.decoder_transition(x)
        x = self.decoder(x)

        return x


class VGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False):
        super(VGGEncoder, self).__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = EncoderBlock(input_dim=3, output_dim=64, hidden_dim=64, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = EncoderBlock(input_dim=64, output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = EncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=enable_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class VGGDecoder(nn.Module):

    def __init__(self, configs, enable_bn=False):
        super(VGGDecoder, self).__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = DecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = DecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = DecoderBlock(input_dim=64, output_dim=3, hidden_dim=64, layers=configs[4], enable_bn=enable_bn)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False, padding=1):

        super(EncoderBlock, self).__init__()

        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn, padding=padding)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn,
                                         padding=padding)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn,
                                         padding=padding)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn,
                                         padding=padding)

                self.add_module('%d EncoderLayer' % i, layer)

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False, padding=1):

        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn,
                                         padding=padding)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn,
                                         padding=padding)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn,
                                         padding=padding)

                self.add_module('%d DecoderLayer' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class EncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn, padding=1):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=padding),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)


class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn, padding=1):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=padding),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=padding),
            )

    def forward(self, x):

        return self.layer(x)


class EncoderTransitionBlock(nn.Module):
    def __init__(self, enable_bn=False):
        super(EncoderTransitionBlock, self).__init__()
        self.conv1 = EncoderLayer(input_dim=512, output_dim=256, enable_bn=enable_bn, padding=1)
        self.conv2 = EncoderLayer(input_dim=256, output_dim=128, enable_bn=enable_bn, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        return x


class DecoderTransitionBlock(nn.Module):
    def __init__(self, enable_bn=False):
        super(DecoderTransitionBlock, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv1 = DecoderLayer(input_dim=256, output_dim=256, enable_bn=enable_bn)
        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv2 = DecoderLayer(input_dim=512, output_dim=512, enable_bn=enable_bn)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)

        return x
# if __name__ == "__main__":

#     input = torch.randn((5,3,224,224))

#     configs = get_configs()

#     model = VGGAutoEncoder(configs)

#     output = model(input)

#     print(output.shape)



