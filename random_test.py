import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json
# 设置随机种子，保证结果可复现
np.random.seed(42)

def generate_lottery_numbers(num_samples):
    """
    生成模拟的双色球历史数据
    参数：
        num_samples: 需要生成的样本数量
    返回：
        DataFrame格式的模拟历史数据
    """
    # 生成红球号码（6个，范围1-33）
    red_balls = np.random.randint(1, 34, size=(num_samples, 6))
    # 对每行进行排序
    red_balls = np.sort(red_balls, axis=1)
    
    # 生成蓝球号码（1个，范围1-16）
    blue_balls = np.random.randint(1, 17, size=(num_samples, 1))
    
    # 合并红蓝球号码
    all_numbers = np.concatenate([red_balls, blue_balls], axis=1)
    
    # 创建日期索引
    dates = pd.date_range(end='2023-12-31', periods=num_samples, freq='3D')
    
    # 创建DataFrame
    columns = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']
    df = pd.DataFrame(all_numbers, columns=columns, index=dates)
    
    return df

def add_features(df):
    """
    添加特征工程
    参数：
        df: 原始DataFrame
    返回：
        添加特征后的DataFrame
    """
    # 复制一份数据
    df_featured = df.copy()
    
    # 计算红球的和值
    df_featured['sum_red'] = df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']].sum(axis=1)
    
    # 计算红球的方差
    df_featured['var_red'] = df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']].var(axis=1)
    
    # 计算红球中奇数的数量
    df_featured['odd_count'] = df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']].apply(
        lambda x: sum(i % 2 == 1 for i in x), axis=1)
    
    # 计算红球的最大间隔
    df_featured['max_gap'] = df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']].apply(
        lambda x: max(np.diff(x)), axis=1)
    
    # 计算红球的最小间隔
    df_featured['min_gap'] = df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']].apply(
        lambda x: min(np.diff(x)), axis=1)
    
    return df_featured

def build_model(input_dim):
    """
    构建神经网络模型
    参数：
        input_dim: 输入特征的维度
    返回：
        编译好的模型
    """
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(7)  # 输出层：6个红球+1个蓝球
    ])
    
    model.compile(optimizer='adam',
                 loss=MeanSquaredError(),
                 metrics=[MeanAbsoluteError()])
    
    return model

def plot_history(history, save_path=None):
    """
    绘制训练历史图
    参数：
        history: 模型训练历史
        save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制MAE曲线
    plt.subplot(1, 2, 2)
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def process_predictions(predictions):
    """
    处理模型预测结果，确保符合双色球规则
    参数：
        predictions: 模型原始预测结果
    返回：
        处理后的预测结果
    """
    # 将预测值限制在合法范围内
    red_balls = np.clip(predictions[:, :6], 1, 33).round().astype(int)
    blue_ball = np.clip(predictions[:, 6:], 1, 16).round().astype(int)
    
    # 确保红球不重复
    for i in range(len(red_balls)):
        while len(set(red_balls[i])) != 6:
            mask = np.random.choice(6, size=2, replace=False)
            red_balls[i][mask[0]] = np.random.randint(1, 34)
            red_balls[i].sort()
    
    return np.concatenate([red_balls, blue_ball], axis=1)

def evaluate_predictions(y_true, y_pred):
    """
    评估预测结果
    参数：
        y_true: 真实值
        y_pred: 预测值
    """
    # 计算整体MAE和RMSE
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算每个球的准确匹配率
    exact_matches = (y_true == y_pred)
    ball_accuracy = exact_matches.mean(axis=0) * 100
    
    # 计算完全匹配的概率（所有球都对）
    perfect_matches = np.all(exact_matches, axis=1)
    perfect_match_rate = perfect_matches.mean() * 100
    
    results = {
        "整体MAE": mae,
        "整体RMSE": rmse,
        "各球准确率": {f"球{i+1}": acc for i, acc in enumerate(ball_accuracy)},
        "完全匹配率": perfect_match_rate
    }
    
    return results

def save_model_and_metadata(model, feature_columns, save_dir='saved_model'):
    """
    保存模型和相关元数据
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存模型
    model.save(os.path.join(save_dir, 'model.h5'))
    
    # 保存特征列名
    with open(os.path.join(save_dir, 'feature_columns.json'), 'w') as f:
        json.dump(list(feature_columns), f)

def load_saved_model(save_dir='saved_model'):
    """
    加载保存的模型和元数据
    """
    # 加载模型
    model = load_model(os.path.join(save_dir, 'model.h5'))
    
    # 加载特征列名
    with open(os.path.join(save_dir, 'feature_columns.json'), 'r') as f:
        feature_columns = json.load(f)
    
    return model, feature_columns

def main():
    # 1. 生成训练数据
    print("生成训练数据...")
    num_samples = 10000  # 增加样本量
    df = generate_lottery_numbers(num_samples)
    print("原始数据示例：")
    print(df.head())
    print("\n数据形状：", df.shape)
    
    # 2. 特征工程
    print("\n进行特征工程...")
    df_featured = add_features(df)
    print("添加特征后的数据示例：")
    print(df_featured.head())
    
    # 3. 准备训练数据
    X = df_featured.values
    y = df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].values
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. 构建和训练模型
    print("\n构建和训练模型...")
    model = build_model(X.shape[1])
    model.summary()
    
    # 6. 训练模型
    history = model.fit(X_train, y_train,
                       epochs=100,  # 增加训练轮数
                       batch_size=32,
                       validation_data=(X_test, y_test),
                       verbose=1)
    
    # 7. 保存训练历史图
    plot_history(history, save_path='training_history.png')
    
    # 8. 保存模型和元数据
    print("\n保存模型...")
    save_model_and_metadata(model, df_featured.columns)
    
    # 9. 生成新的测试数据
    print("\n生成新的测试数据进行评估...")
    test_samples = 1000
    df_test = generate_lottery_numbers(test_samples)
    df_test_featured = add_features(df_test)
    
    # 10. 加载保存的模型
    print("\n加载保存的模型...")
    loaded_model, feature_columns = load_saved_model()
    
    # 11. 使用加载的模型进行预测
    X_new = df_test_featured.values
    y_new = df_test[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].values
    
    predictions = loaded_model.predict(X_new)
    processed_predictions = process_predictions(predictions)
    
    # 12. 评估结果
    print("\n评估预测结果...")
    evaluation_results = evaluate_predictions(y_new, processed_predictions)
    
    # 13. 输出评估结果
    print("\n模型评估结果：")
    print(json.dumps(evaluation_results, indent=2))
    
    # 14. 展示一些预测示例
    print("\n预测示例：")
    for i in range(5):
        print(f"\n预测 {i+1}:")
        print("预测号码:", processed_predictions[i])
        print("实际号码:", y_new[i])

if __name__ == "__main__":
    main()
