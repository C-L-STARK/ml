import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate
from typing import Dict, List
import os

class CIFAR10Dataset(torch.utils.data.Dataset):
    """
    自定义CIFAR10数据集类，适配ViT模型的输入要求
    """
    def __init__(self, is_train: bool, processor: ViTImageProcessor):
        """
        初始化数据集
        Args:
            is_train (bool): 是否为训练集
            processor (ViTImageProcessor): ViT图像处理器
        """
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=is_train,
            download=True
        )
        self.processor = processor
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个数据样本
        Args:
            idx (int): 索引
        Returns:
            Dict: 包含处理后的图像和标签
        """
        image, label = self.dataset[idx]
        # 处理图像以适应ViT的输入要求
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': label
        }

class ModelTrainer:
    """
    模型训练器类，整合了模型训练的所有功能
    """
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        初始化训练器
        Args:
            model_name (str): 预训练模型名称
            batch_size (int): 批次大小
            num_epochs (int): 训练轮数
            learning_rate (float): 学习率
            checkpoint_dir (str): 模型保存目录
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 初始化评估指标
        self.metric = evaluate.load("accuracy")
        
        # 初始化TensorBoard
        self.writer = SummaryWriter('runs/cifar10_vit_experiment')
        
        print(f"Using device: {self.device}")
    
    def prepare_model(self):
        """
        准备模型和处理器
        """
        print("Loading pre-trained model...")
        # 加载预训练模型
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=10,  # CIFAR10的类别数
            ignore_mismatched_sizes=True  # 允许改变分类头的大小
        )
        
        # 加载图像处理器
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        
        # 将模型移到指定设备
        self.model = self.model.to(self.device)
        print("Model prepared successfully!")
    
    def prepare_data(self):
        """
        准备数据集和数据加载器
        """
        print("Preparing datasets...")
        # 创建训练集和测试集
        train_dataset = CIFAR10Dataset(True, self.processor)
        test_dataset = CIFAR10Dataset(False, self.processor)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        print(f"Datasets prepared! Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")
    
    def train_step(self, batch: Dict) -> float:
        """
        单个训练步骤
        Args:
            batch (Dict): 包含输入数据的字典
        Returns:
            float: 训练损失
        """
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        outputs = self.model(**batch)
        loss = outputs.loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self) -> float:
        """
        评估模型
        Returns:
            float: 评估准确率
        """
        self.model.eval()
        total_accuracy = 0
        total_count = 0
        
        print("\nEvaluating model...")
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(-1)
                accuracy = (predictions == batch['labels']).float().mean()
                total_accuracy += accuracy.item() * len(batch['labels'])
                total_count += len(batch['labels'])
        
        final_accuracy = total_accuracy / total_count
        print(f"Evaluation accuracy: {final_accuracy:.4f}")
        return final_accuracy
    
    def save_checkpoint(self, accuracy: float, epoch: int):
        """
        保存模型检查点
        Args:
            accuracy (float): 当前准确率
            epoch (int): 当前轮数
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'model_epoch_{epoch}_acc_{accuracy:.4f}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """
        完整的训练流程
        """
        print("\nStarting training...")
        # 准备优化器和学习率调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # 训练循环
        global_step = 0
        best_accuracy = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            epoch_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_loss += loss
                global_step += 1
                
                # 记录训练指标
                self.writer.add_scalar('Training/Loss', loss, global_step)
                
                # 打印进度
                if (batch_idx + 1) % 50 == 0:
                    print(f"Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss:.4f}")
            
            # 计算平均损失
            avg_loss = epoch_loss / len(self.train_loader)
            
            # 评估模型
            accuracy = self.evaluate()
            self.writer.add_scalar('Training/Accuracy', accuracy, epoch)
            self.writer.add_scalar('Training/AvgLoss', avg_loss, epoch)
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(accuracy, epoch)
            
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'Average Loss: {avg_loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Best Accuracy: {best_accuracy:.4f}')
            print('-' * 50)
        
        self.writer.close()
        print(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
        return best_accuracy

def main():
    """
    主函数，运行完整的训练流程
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建训练器实例
    trainer = ModelTrainer(
        model_name="google/vit-base-patch16-224",
        batch_size=32,
        num_epochs=10,
        learning_rate=2e-5
    )
    
    # 准备模型和数据
    trainer.prepare_model()
    trainer.prepare_data()
    
    # 开始训练
    best_accuracy = trainer.train()
    print(f'Training completed! Best accuracy: {best_accuracy:.4f}')

if __name__ == "__main__":
    main()
