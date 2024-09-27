import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 注意这里的改动
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleCNN()

# 设定输入数据的维度（例如：3个颜色通道，32x32像素的图像）
input_size = (3, 32, 32)

# 计算 FLOPs 和参数数量
with torch.cuda.device(0):  # 指定 GPU 设备
    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True, verbose=True)

print(f"FLOPs: {flops}")
print(f"Params: {params}")

input_tensor = torch.randn(1, 3, 32, 32)  # 符合模型输入需求的张量

flops = FlopCountAnalysis(model, input_tensor)  # 确保这里的 input_tensor 适合您的模型
print(f"Total FLOPs: {flops.total()}")

