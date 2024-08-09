import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from textdata import ChromosomeDataset


# 假设 X 是你的输入数据，y 是标签
# X = np.random.randint(0, 10, size=(1000, 30000))  # 示例数据
# y = np.random.randint(0, 2, size=(1000,))  # 示例标签

# 数据集类
class DNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 模型定义
class DNAClassifier(nn.Module):
    def __init__(self, input_size):
        super(DNAClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(loss)


# 评估函数
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


# 主函数
def main():
    # 数据分割
    tsv_dir = '/home/siweideng/OxTium_cfDNA'
    dataset = ChromosomeDataset(tsv_dir)
    X = dataset.processed_data
    y = dataset.labels
    X_train, X_test, y_train, y_test = train_test_split(X / 100, y, test_size=0.4, random_state=42)

    # 创建数据加载器
    train_dataset = DNADataset(X_train, y_train)
    test_dataset = DNADataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = DNAClassifier(input_size=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} completed")

    # 评估模型
    preds, labels = evaluate(model, test_loader, device)

    # 计算 AUC
    auc = roc_auc_score(labels, preds)
    print(f"AUC: {auc:.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds > 0.5)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()