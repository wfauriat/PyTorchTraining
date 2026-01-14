import torch.nn as nn
import torch.nn.functional as F

from typing import List

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture similar to LeNet.
    Architecture: Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pools of 2x2 on 32x32 input: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten
        x = x.flatten(1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class VGGStyleCNN(nn.Module):
    """
    VGG-style CNN with batch normalization.
    Uses repeated 3x3 convolutions with pooling.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Block 1: 2 convs, pool
        self.block1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )
        
        # Block 2: 2 convs, pool
        self.block2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2)  # 16 -> 8
        )
        
        # Block 3: 3 convs, pool
        self.block3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2)  # 8 -> 4
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x




class BasicBlock(nn.Module):
    """
    Basic ResNet block with skip connection.
    
    Structure:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Need to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for deeper ResNets (50, 101, 152).
    
    Structure:
        x -> 1x1 Conv -> 3x3 Conv -> 1x1 Conv -> (+x) -> ReLU
    
    The 1x1 convs reduce then expand channels (bottleneck).
    """
    
    expansion = 4  # Output channels = out_channels * 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv (main processing)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet implementation.
    
    Configurations:
        ResNet-18: [2, 2, 2, 2] with BasicBlock
        ResNet-34: [3, 4, 6, 3] with BasicBlock
        ResNet-50: [3, 4, 6, 3] with Bottleneck
    """
    
    def __init__(self, block, num_blocks: List[int], num_classes: int = 10):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int):
        """Create a layer with multiple residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classifier
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x