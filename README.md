# Japanese Name Generator with Transformer

基于 Transformer 的日语人名生成模型，可以根据性别和前缀生成日语假名人名。

## 项目简介

本项目使用深度学习技术训练一个 Transformer 模型，用于生成符合日语命名规则的假名人名。模型支持：
- 根据性别（男/女）生成相应风格的名字
- 根据给定前缀续写完整名字
- 字符级别的序列生成

## 功能特点

- 🚀 Transformer模型架构
- 📊 完整的训练可视化（Loss 曲线、准确率曲线）
- 💾 自动保存最佳模型
- 🎯 测试集验证
- 📈 Embedding 可视化

## 环境要求

```bash
Python 3.8+
torch>=2.0.0
pandas
matplotlib
scikit-learn
transformers
```

## 安装依赖

```bash
pip install torch pandas matplotlib scikit-learn transformers
```

## 数据集

数据集应包含以下文件，放在 `data/` 目录下：
- `gendec-train.csv` - 训练集
- `gendec-test.csv` - 测试集
- `gendec-dev.csv` - 验证集（可选）

数据格式示例：
```csv
Gender,Hiragana
Male,たろう
Female,はなこ
```

## 使用方法

### 1. 训练模型

打开 `train.ipynb` 笔记本，按顺序运行单元格：

```python
# 初始化模型
model = TransformerModel(
    vocab_size=len(hiragana_map), 
    embedding_dim=256,
    dropout=0.2,
    num_heads=8,
    layers=6
)

# 训练
train_losses, test_losses, test_accs = train(
    model, 
    transinput, 
    test_input=test_transinput,
    epochs=100, 
    lr=0.0001, 
    batch_size=128,
    save_path='best_model.pth'
)
```

### 2. 加载模型

```python
model = TransformerModel(vocab_size=len(hiragana_map), embedding_dim=256)
model = load_model(model, 'best_model.pth')
```

### 3. 生成名字

```python
sex = "Female"  # 或 "Male"
prefix = "さくら"  # 名字前缀

# 生成完整名字
test_in = ["."] + [sex] + [i for i in prefix]
test_x = [hiragana_map.get(c, 0) for c in test_in]

model.eval()
with torch.no_grad():
    while True:
        test_input = torch.tensor(test_x).unsqueeze(0).to(device)
        output = model(test_input)
        output_index = output[0, -1].argmax().item()
        test_x.append(output_index)
        if output_index == 0 or len(test_x) > padding_length:
            break

result = "".join([hiragana_index_map.get(i, "") for i in test_x[1:-1]])
print(result)
```

## 模型架构

### Transformer Model
- **Embedding**: 256维字符嵌入 + 位置编码
- **Multi-Head Attention**: 8个注意力头
- **Layers**: 6层 Transformer Block
- **Dropout**: 0.2


### 训练细节
- **优化器**: Adam (lr=0.0001)
- **Batch Size**: 128
- **梯度裁剪**: max_norm=1.0
- **训练集大小**: ~40,000 样本
- **测试集大小**: ~10,000 样本

## 性能指标

最佳模型性能（100 epochs）：
- **训练 Loss**: 1.44
- **测试 Loss**: 1.65
- **测试准确率**: ~45%

## 文件结构

```
JapaneseNames/
├── train.ipynb           # 主训练笔记本
├── data/                 # 数据集目录
│   ├── gendec-train.csv
│   ├── gendec-test.csv
│   └── gendec-dev.csv
├── best_model.pth        # 最佳模型（训练过程中自动保存）
├── best_model_final.pth  # 最终模型
├── .gitignore
└── README.md
```


## 可视化

项目包含多种可视化功能：
- 训练/测试 Loss 曲线
- 测试准确率曲线
- Embedding 2D 投影（PCA降维）

## 注意事项

1. 模型文件（*.pth）较大，已在 .gitignore 中排除
2. 训练需要 GPU 支持以获得较快速度
3. 数据集可能包含个人信息，请注意隐私保护

## License

MIT License


