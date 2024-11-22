import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 创建简单数据集
def create_dataset():
    X = torch.linspace(-5, 5, 200).reshape(-1, 1)
    y = 0.2 * X + torch.sin(X) + torch.randn_like(X) * 0.1
    return X, y

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# 训练函数
def train_model(model, X, y, optimizer, epochs=1000):
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
            
    return losses

# 可视化权重分布
def plot_weights(model, title):
    weights = []
    for param in model.parameters():
        if len(param.data.size()) > 1:  # 只收集权重矩阵，不包括偏置
            weights.extend(param.data.cpu().numpy().flatten())
    
    plt.hist(weights, bins=50, density=True)
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')

# 主函数
def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建数据
    X, y = create_dataset()
    
    # 创建两个相同的模型
    model_no_decay = SimpleNet()
    model_with_decay = SimpleNet()
    
    # 两个优化器，一个有权重衰减，一个没有
    optimizer_no_decay = torch.optim.Adam(model_no_decay.parameters(), lr=0.01)
    optimizer_with_decay = torch.optim.Adam(model_with_decay.parameters(), 
                                          lr=0.01, 
                                          weight_decay=0.01)
    
    # 训练模型
    print("Training model without weight decay...")
    losses_no_decay = train_model(model_no_decay, X, y, optimizer_no_decay)
    
    print("\nTraining model with weight decay...")
    losses_with_decay = train_model(model_with_decay, X, y, optimizer_with_decay)
    
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    # 损失曲线对比
    plt.subplot(1, 3, 1)
    plt.plot(losses_no_decay, label='No Weight Decay')
    plt.plot(losses_with_decay, label='With Weight Decay')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 权重分布对比
    plt.subplot(1, 3, 2)
    plot_weights(model_no_decay, 'Weight Distribution (No Decay)')
    
    plt.subplot(1, 3, 3)
    plot_weights(model_with_decay, 'Weight Distribution (With Decay)')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
