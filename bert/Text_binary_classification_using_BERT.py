import pandas as pd
import numpy as np
import random
import time
import math
from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Sequential, Linear, ReLU, Sigmoid
from mlx.nn.losses import binary_cross_entropy
from mlx.optimizers import Adam
from transformers import AutoTokenizer, AutoConfig  # 分词器
from model.bert_by_mlx import Bert


class Data:
    def __init__(self):
        self.train_path = 'data/data_set/train.csv'
        self.test_path = 'data/data_set/testB.csv'
        self.train = None
        self.test = None
        self.validate = None
        self.read_data()
        self.process_data()
        self.sample_validate_from_train()

    def read_data(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

    def process_data(self):
        self.train['title'] = self.train['title'].fillna('')
        self.train['abstract'] = self.train['abstract'].fillna('')

        self.test['title'] = self.test['title'].fillna('')
        self.test['abstract'] = self.test['abstract'].fillna('')

        self.train['text'] = (self.train['title'].fillna('') + ' ' + self.train['author'].fillna('') + ' ' +
                              self.train['abstract'].fillna('') + ' ' + self.train['Keywords'].fillna(''))
        self.test['text'] = (self.test['title'].fillna('') + ' ' + self.test['author'].fillna('') + ' ' +
                             self.test['abstract'].fillna(''))
        self.test['Keywords'] = self.test['title'].fillna('')

    def sample_validate_from_train(self):
        # 取多少训练集的数据作为验证集
        validation_ratio = 0.1
        self.validate = self.train.sample(frac=validation_ratio)
        self.train = self.train[~self.train.index.isin(self.validate.index)]


class MyDataset:
    def __init__(self, mode='train', data=Data()):
        self.mode = mode
        # 根据 mode 获取对应的数据集
        if mode == 'train':
            self.dataset = data.train
        elif mode == 'validation':
            self.dataset = data.validate
        elif mode == 'test':
            self.dataset = data.test
        else:
            raise Exception('Unknown mode {}'.format(mode))

    def __getitem__(self, index):
        # 获取指定索引的数据
        data = self.dataset.iloc[index]
        text = data['text']  # 文本数据
        if self.mode == 'test':
            label = data['uuid']  # 测试集时使用 uuid 作为标签
        else:
            label = data['label']  # 训练集和验证集使用 label 作为标签
        return text, label

    def __len__(self):
        # 返回数据集的大小
        return len(self.dataset)


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, collate_fn=None):
        """
        初始化 DataLoader
        :param dataset: 自定义数据集对象，必须实现 __getitem__ 和 __len__
        :param batch_size: 每个 batch 的大小
        :param shuffle: 是否对数据进行随机打乱
        :param collate_fn: 用于处理每个 batch 的自定义函数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        """
        返回一个迭代器，生成批量数据
        """
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)  # 随机打乱索引

        self.current_idx = 0  # 当前索引
        return self

    def __next__(self):
        """
        返回下一个批次数据
        """
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        # 获取当前批次的索引
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        # 根据索引提取数据
        batch = [self.dataset[i] for i in batch_indices]

        # 使用 collate_fn 处理批量数据（如果提供了）
        if self.collate_fn:
            return self.collate_fn(batch)
        else:
            return batch

    def __len__(self):
        """
        返回批次数量
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果
    """
    # 将输入的文本和标签拆分
    text, label = zip(*batch)
    text, label = list(text), list(label)

    # 使用 tokenizer 对文本进行处理
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # padding='max_length' 不够长度的进行填充，truncation=True 长度过长的进行裁剪
    text_max_length = 128  # 设定最大文本长度
    # 返回字典格式的输出
    src = tokenizer(text, padding='max_length', max_length=text_max_length, return_tensors='pt', truncation=True)

    # 处理后的文本是字典，包含了输入ID、注意力掩码等，转换为 MLX 张量类型
    input_ids = mx.array(src['input_ids'].numpy())
    attention_mask = mx.array(src['attention_mask'].numpy())

    # 将标签转换为长整型的张量
    label = np.array(label, dtype=np.int32)
    target = mx.array(label)

    # 返回处理后的数据
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, target


class MyModel(nn.Module):
    # 定义预测模型，该模型由bert模型加上最后的预测层组成
    def __init__(self):
        super(MyModel, self).__init__()
        config = AutoConfig.from_pretrained('bert-base-uncased', mirror='tuna')
        self.bert = Bert(config)
        # 初始化 Sequential 模块
        linear = Linear(768, 256)
        relu = ReLU()
        linear2 = Linear(256, 1)
        sigmoid = Sigmoid()
        self.predictor = Sequential(linear, relu, linear2, sigmoid)

    def __call__(self, src):
        outputs = self.bert(**src)[0][:, 0, :]
        return self.predictor(outputs)


def validate():
    model.eval()
    total_loss = 0.
    total_correct = 0
    for iteration, (inputs_, targets_) in enumerate(train_loader):
        outputs = model(inputs_)
        # 损失函数
        loss = binary_cross_entropy(outputs.reshape(-1), targets_)
        total_loss += mx.mean(loss)

        # Step 1: 逻辑判断
        predictions = mx.array(outputs >= 0.5)  # 布尔张量，元素为 True 或 False

        # Step 2: 转换为浮点型并拉平
        predictions = predictions.astype(mx.float32).flatten()

        # Step 3: 比较预测值和目标值
        correct_predictions = predictions == targets_.astype(mx.float32).flatten()

        # Step 4: 计算正确预测的数量
        correct_num = mx.sum(correct_predictions.astype(mx.float32))
        total_correct += correct_num

    return total_correct / len(validation_dataset), total_loss / len(validation_dataset)


def loss_func(model, inputs, targets, reduce=True):
    logits = model(inputs)
    losses = binary_cross_entropy(logits.reshape(-1), targets)
    return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


batch_size = 16
epochs = 188
lr = 6e-6
total_loss = 0.  # 定义几个变量，帮助打印loss
step_num = 0  # 记录步数
log_per_step = 20  # 每多少步打印一次loss
best_accuracy = 0  # 记录在验证集上最好的准确率
train_dataset = MyDataset('train')
validation_dataset = MyDataset('validation')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 定义优化器
optimizer = Adam(learning_rate=lr)
mx.set_default_device(mx.gpu)

# 初始化模型
model = MyModel()
mx.eval(model.parameters())
state = [model.state, optimizer.state]


@partial(mx.compile, inputs=state, outputs=state)
def step(inputs, targets):
    # 计算损失、梯度
    loss_and_grad_fn = nn.value_and_grad(model, loss_func)
    loss, grads = loss_and_grad_fn(model, inputs, targets)
    optimizer.update(model, grads)
    return loss


tic = time.perf_counter()
# 开始训练
for epoch in range(epochs):
    model.train()
    losses = []
    for i, (inputs, targets) in enumerate(train_loader):
        loss = step(inputs, targets)
        mx.eval(state)
        losses.append(loss.item())
        if (i + 1) % log_per_step == 0:
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            print(
                f"Epoch {epoch+1}/{epochs}, Step: {i+1}/{len(train_loader)}, "
                f"Train loss {train_loss:.3f}, "
                f"It/sec {log_per_step / (toc - tic):.3f}"
            )
            tic = time.perf_counter()
            losses = []
    # 一个epoch后，使用过验证集进行验证
    accuracy, validation_loss = validate()
    print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, accuracy, validation_loss))