import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(0)
np.random.seed(0)

# 定义超参数
input_size = 1    # 输入特征维度
hidden_size = 16  # 隐藏层神经元数量
output_size = 1   # 输出特征维度
num_layers = 2    # RNN层数
seq_length = 5    # 序列长度
learning_rate = 0.01
num_epochs = 100

# 生成一些简单的时间序列数据
def generate_data():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    return y

# 创建数据集
data = generate_data()
x_train = []
y_train = []
for i in range(len(data) - seq_length):
    x_train.append(data[i:i+seq_length])
    y_train.append(data[i+seq_length])
x_train = np.array(x_train)
y_train = np.array(y_train)

# 转换为张量
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型、损失函数和优化器
model = RNNModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 初始化TensorBoard
writer = SummaryWriter('runs/rnn_experiment')

# 训练模型
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # 记录损失值到TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 关闭TensorBoard写入器
writer.close()
