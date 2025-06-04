import torch
import torch.nn as nn
import math

class ResNet3DBuilder:
    @staticmethod
    def _conv_bn_relu3D(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), padding='same'):
        if padding == 'same':
            padding = tuple(k // 2 for k in kernel_size)
            
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _bn_relu_conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), padding='same'):
        if padding == 'same':
            padding = tuple(k // 2 for k in kernel_size)
            
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=False)
        )

    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=(1,1,1), is_first_block_of_first_layer=False):
            super().__init__()
            
            if is_first_block_of_first_layer:
                self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                                     stride=stride, padding=1, bias=False)
            else:
                self.conv1 = ResNet3DBuilder._bn_relu_conv3d(
                    in_channels, out_channels, kernel_size=(3,3,3), stride=stride)
                
            self.conv2 = ResNet3DBuilder._bn_relu_conv3d(
                out_channels, out_channels, kernel_size=(3,3,3))
            
            if stride != (1,1,1) or in_channels != out_channels:
                self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                        stride=stride, bias=False)
            else:
                self.shortcut = None
                
        def forward(self, x):
            residual = x
            
            out = self.conv1(x)
            out = self.conv2(out)
            
            if self.shortcut is not None:
                residual = self.shortcut(x)
                
            out += residual
            return out

    class BottleneckBlock(nn.Module):
        expansion = 4
        
        def __init__(self, in_channels, out_channels, stride=(1,1,1), is_first_block_of_first_layer=False):
            super().__init__()
            
            if is_first_block_of_first_layer:
                self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                     stride=stride, padding=0, bias=False)
            else:
                self.conv1 = ResNet3DBuilder._bn_relu_conv3d(
                    in_channels, out_channels, kernel_size=(1,1,1), stride=stride)
                
            self.conv2 = ResNet3DBuilder._bn_relu_conv3d(
                out_channels, out_channels, kernel_size=(3,3,3))
            self.conv3 = ResNet3DBuilder._bn_relu_conv3d(
                out_channels, out_channels * self.expansion, kernel_size=(1,1,1))
            
            if stride != (1,1,1) or in_channels != out_channels * self.expansion:
                self.shortcut = nn.Conv3d(in_channels, out_channels * self.expansion,
                                        kernel_size=1, stride=stride, bias=False)
            else:
                self.shortcut = None
                
        def forward(self, x):
            residual = x
            
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            
            if self.shortcut is not None:
                residual = self.shortcut(x)
                
            out += residual
            return out

    class ResNet3D(nn.Module):
        def __init__(self, input_shape, num_outputs, block_fn, repetitions):
            super().__init__()
            
            self.in_channels = 32
            
            self.conv1 = ResNet3DBuilder._conv_bn_relu3D(
                input_shape[0], self.in_channels, kernel_size=(7,7,3), stride=(2,2,1))
            
            self.pool1 = nn.MaxPool3d(kernel_size=3, stride=(2,2,1), padding=1)
            
            self.conv2 = ResNet3DBuilder._conv_bn_relu3D(
                self.in_channels, self.in_channels, kernel_size=(3,3,3), stride=(2,2,2))
            
            # Residual blocks
            self.layer1 = self._make_layer(block_fn, 32, repetitions[0], is_first_layer=True)
            self.layer2 = self._make_layer(block_fn, 64, repetitions[1])
            self.layer3 = self._make_layer(block_fn, 128, repetitions[2])
            self.layer4 = self._make_layer(block_fn, 256, repetitions[3])
            
            # Final layers
            self.bn_relu = nn.Sequential(
                nn.BatchNorm3d(self.in_channels),
                nn.ReLU(inplace=True)
            )
            
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            
            self.fc = nn.Linear(self.in_channels, num_outputs)
            
#             if num_outputs == 1:
#                 self.final_activation = nn.Sigmoid()
#             else:
#                 self.final_activation = nn.Softmax(dim=1)
            
            self._initialize_weights()
            
        def _make_layer(self, block_fn, out_channels, blocks, is_first_layer=False):
            layers = []
            stride = (2,2,2) if not is_first_layer else (1,1,1)
            
            layers.append(block_fn(self.in_channels, out_channels, stride=stride,
                                 is_first_block_of_first_layer=(is_first_layer and len(layers)==0)))
            
            self.in_channels = out_channels * (block_fn.expansion if hasattr(block_fn, 'expansion') else 1)
            
            for _ in range(1, blocks):
                layers.append(block_fn(self.in_channels, out_channels))
                
            return nn.Sequential(*layers)
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm3d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    
        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.bn_relu(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            # x = self.final_activation(x)
            
            return x

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        return ResNet3DBuilder.ResNet3D(input_shape, num_outputs, block_fn, repetitions)
    
    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResNet3DBuilder.build(input_shape, num_outputs, 
                                   ResNet3DBuilder.BasicBlock, [2, 2, 2, 2])
    
    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResNet3DBuilder.build(input_shape, num_outputs,
                                   ResNet3DBuilder.BasicBlock, [3, 4, 6, 3])
    
    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResNet3DBuilder.build(input_shape, num_outputs,
                                   ResNet3DBuilder.BottleneckBlock, [3, 4, 6, 3])
    
    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResNet3DBuilder.build(input_shape, num_outputs,
                                   ResNet3DBuilder.BottleneckBlock, [3, 4, 23, 3])
    
    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResNet3DBuilder.build(input_shape, num_outputs,
                                   ResNet3DBuilder.BottleneckBlock, [3, 8, 36, 3])

def create_model(input_shape=(1, 352, 352, 35), num_outputs=1):
    model = ResNet3DBuilder.build_resnet_18(input_shape, num_outputs)
    return model



