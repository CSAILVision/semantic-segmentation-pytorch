import torch
import torch.nn as nn
import torchvision


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='vgg16_dilated', fc_dim=1024, weights=''):
        if arch == 'vgg16_dilated':
            original_vgg = torchvision.models.vgg16(pretrained=True)
            conv5 = (24, 26, 28)
            pool4n5 = (23, 30)
            net_encoder = VggDilated(original_vgg,
                                     conv5,
                                     pool4n5,
                                     dropout2d=True)
        elif arch == 'vgg19_dilated':
            original_vgg = torchvision.models.vgg19(pretrained=True)
            conv5 = (28, 30, 32, 34)
            pool4n5 = (27, 36)
            net_encoder = VggDilated(original_vgg,
                                     conv5,
                                     pool4n5,
                                     dropout2d=True)
        elif arch == 'resnet34_dilated8':
            original_resnet = torchvision.models.resnet34(pretrained=True)
            net_encoder = ResnetDilated(original_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            original_resnet = torchvision.models.resnet34(pretrained=True)
            net_encoder = ResnetDilated(original_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            original_resnet = torchvision.models.resnet50(pretrained=True)
            net_encoder = Resnet(original_resnet)
        elif arch == 'resnet50_dilated8':
            original_resnet = torchvision.models.resnet50(pretrained=True)
            net_encoder = ResnetDilated(original_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            original_resnet = torchvision.models.resnet50(pretrained=True)
            net_encoder = ResnetDilated(original_resnet,
                                        dilate_scale=16)
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(torch.load(weights))
        return net_encoder

    def build_decoder(self, arch='c1bilinear', fc_dim=1024, num_class=150,
                      segSize=384, weights='', use_softmax=False):
        if arch == 'c1bilinear':
            net_decoder = C1Bilinear(num_class=num_class,
                                     fc_dim=fc_dim,
                                     segSize=segSize,
                                     use_softmax=use_softmax)
        elif arch == 'c5bilinear':
            net_decoder = C5Bilinear(num_class=num_class,
                                     fc_dim=fc_dim,
                                     segSize=segSize,
                                     use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(torch.load(weights))
        return net_decoder


class VggDilated(nn.Module):
    def __init__(self, original_vgg, conv5, pool4n5, dropout2d=True):
        super(VggDilated, self).__init__()

        # make conv5 dilated
        for i in conv5:
            original_vgg.features[i].dilation = (2, 2)
            original_vgg.features[i].padding = (2, 2)
        # take away pool4 and pool5
        modules = [x for i, x in enumerate(original_vgg.features)
                   if i not in pool4n5]
        self.features = nn.Sequential(*(modules))

        # convert fc weights into conv1x1 weights
        self.conv6 = nn.Conv2d(512, 4096, 7, 1, 12, 4)
        self.conv6.weight.data.copy_(
            original_vgg.classifier[0].weight.data.resize_(4096, 512, 7, 7))
        self.conv6.bias.data.copy_(original_vgg.classifier[0].bias.data)
        self.conv7 = nn.Conv2d(4096, 4096, 1, 1, 0)
        self.conv7.weight.data.copy_(
            original_vgg.classifier[3].weight.data.resize_(4096, 4096, 1, 1))
        self.conv7.bias.data.copy_(original_vgg.classifier[3].bias.data)

        self.relu6 = nn.ReLU(True)
        self.relu7 = nn.ReLU(True)

        if dropout2d:
            self.dropout6 = nn.Dropout2d(0.5)
            self.dropout7 = nn.Dropout2d(0.5)
        else:
            self.dropout6 = nn.Dropout(0.5)
            self.dropout7 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout6(self.relu6(self.conv6(x)))
        x = self.dropout7(self.relu7(self.conv7(x)))

        return x


class Resnet(nn.Module):
    def __init__(self, original_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, take away AvgPool and FC
        self.features = nn.Sequential(*list(original_resnet.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


class ResnetDilated(nn.Module):
    def __init__(self, original_resnet, dilate_scale=8, dropout2d=False):
        super(ResnetDilated, self).__init__()
        self.dropout2d = dropout2d
        from functools import partial

        if dilate_scale == 8:
            original_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            original_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            original_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, take away AvgPool and FC
        self.features = nn.Sequential(*list(original_resnet.children())[:-2])

        if self.dropout2d:
            self.dropout = nn.Dropout2d(0.5)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.features(x)
        if self.dropout2d:
            x = self.dropout(x)
        return x


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        x = self.conv_last(x)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x


# 2 conv with dilation=2, 2 conv with dilation=1, last conv, bilinear upsample
class C5Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(C5Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        # convs, dilation=2
        self.conv1 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        self.conv2 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        # convs, dilation=1
        self.conv3 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        self.conv4 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(fc_dim, momentum=0.1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.conv_last(x)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x
