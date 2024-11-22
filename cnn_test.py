import torch
import torchvision
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class TestModel:
    def __init__(self, checkpoint_path, model_name="google/vit-base-patch16-224"):
        """
        初始化测试类
        Args:
            checkpoint_path (str): 模型检查点路径
            model_name (str): 原始预训练模型名称
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 设备设置，Apple M系列使用 mps 
        self.checkpoint_path = checkpoint_path  # 检查点：保存了权重、模型结构
        self.model_name = model_name # 预训练模型名称
        self.batch_size = 32 # 批次大小（每个批次包含的样本数量）
        
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """
        加载模型和检查点
        """
        print("Loading model...")
        # 初始化模型
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=10, # 最终通道数需要和数据集保持一致
            ignore_mismatched_sizes=True # 忽略
        )
        
        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict']) # 加载保存的状态，训练期间还需要加载 optimize epoch loss 参数供继续训练
        self.model = self.model.to(self.device) # 加载到设备

        # 可选模型保存

        # 切换为执行模式（train为训练模式）
        self.model.eval()
        
        # 加载处理器
        self.processor = ViTImageProcessor.from_pretrained(self.model_name) # 数据处理器
        print("Model loaded successfully!")
        
    def prepare_test_data(self):
        """
        准备测试数据集
        """
        print("Preparing test dataset...")
        self.test_dataset = CIFAR10Dataset(is_train=False, processor=self.processor)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False, # 是否随机打乱
            num_workers=4 # 并行加载
        )
        print(f"Test dataset prepared! Total samples: {len(self.test_dataset)}")
        
    def test(self):
        """
        在测试集上评估模型
        """
        print("\nStarting evaluation...")
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        # torch.no_grad()的作用
        # 1. 禁用梯度计算，减少内存使用
        # 2. 加快前向传播速度
        # 3. 测试时不需要计算梯度
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                probs = F.softmax(outputs, dim=1)
                # 最大概率值即为置信度
                confidence, predicted = torch.max(probs, dim=1)
                print(f"\confidence: {confidence}")
                print(f"\predicted: {predicted}")

                predictions = outputs.logits.argmax(-1)
                
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        accuracy = correct / total
        print(f"\nTest Results:")
        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy, all_predictions, all_labels

# CIFAR10数据集类（与之前的代码相同）
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, is_train: bool, processor: ViTImageProcessor):
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=is_train,
            download=True
        )
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True
        )
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': label
        }

def main():
    # 设置模型检查点路径
    checkpoint_path = "checkpoints/model_epoch_1_acc_0.9827.pth"
    
    # 创建测试器实例
    tester = TestModel(checkpoint_path)
    
    # 加载模型和准备数据
    tester.load_model()
    tester.prepare_test_data()
    
    # 进行测试
    accuracy, predictions, labels = tester.test()
    
    # 输出分类报告
    from sklearn.metrics import classification_report
    target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=target_names))

if __name__ == "__main__":
    main()
