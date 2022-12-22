#!/usr/bin/env python
# coding: utf-8

# # 天气以及时间分类
# 
# ## 赛题名称
# 
# 天气以及时间分类
# 
# https://www.datafountain.cn/competitions/555
# 
# ## 赛题背景
# 在自动驾驶场景中，天气和时间（黎明、早上、下午、黄昏、夜晚）会对传感器的精度造成影响，比如雨天和夜晚会对视觉传感器的精度造成很大的影响。此赛题旨在使用Oneflow框架对拍摄的照片天气和时间进行分类，从而在不同的天气和时间使用不同的自动驾驶策略。
# 
# ## 赛题任务
# 此赛题的数据集由云测数据提供。比赛数据集中包含3000张真实场景下行车记录仪采集的图片，其中训练集包含2600张带有天气和时间类别标签的图片，测试集包含400张不带有标签的图片。
# 
# 本赛题的数据集包含2600张人工标注的天气和时间标签。天气类别包含多云、晴天、雨天、雪天和雾天5个类别；时间包含黎明、早上、下午、黄昏、夜晚5个类别。
# 部分数据可视化及标签如下：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/309f30dcbf6e451ebbab56eb9325cc5d5421dd7ad0f24012adf4e507a04d3482)
# ![](https://ai-studio-static-online.cdn.bcebos.com/0eb6e239e3244c5eb059d3d44d77f880936b68902462439982d158f40068d90c)
# 
# ## 数据说明
# 
# 数据集包含anno和image两个文件夹，anno文件夹中包含2600个标签json文件，image文件夹中包含3000张行车记录仪拍摄的JPEG编码照片。图片标签将字典以json格式序列化进行保存：
# 
# | 列名    | 取值范围                     | 作用         |
# | ------- | ---------------------------- | ------------ |
# | Period  | 黎明、早上、下午、黄昏、夜晚 | 图片拍摄时间 |
# | Weather | 多云、晴天、雨天、雪天、雾天 | 图片天气     |
# 

# In[ ]:


# !echo y | unzip test_dataset.zip > log.log
# !echo y | unzip train_dataset.zip > log.log


# In[ ]:


import io
import math, json
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

paddle.__version__


# # 数据读取

# In[ ]:


train_json = pd.read_json('train.json')  # 读取train_json文件内容，创建train_json对象
train_json['filename'] = train_json['annotations'].apply(lambda x: x['filename'].replace('\\', '/'))  # 将train_json文件中的图片路径名进行规范化，便于访问
train_json['period'] = train_json['annotations'].apply(lambda x: x['period'])  # 获取图片对应的时段标签
train_json['weather'] = train_json['annotations'].apply(lambda x: x['weather']) # 获取图片对用的天气标签

train_json.head() # 用于显示train_json的结构与部分内容


# ## 标签处理

# In[ ]:


train_json['period'], period_dict = pd.factorize(train_json['period'])   # 把常见的字符型变量分解为数字，这里将'period'标签中的各类时段样本用数字代替
train_json['weather'], weather_dict = pd.factorize(train_json['weather'])   # 把常见的字符型变量分解为数字，这里将'whether'标签中的各类天气样本用数字代替


# ## 统计标签

# In[ ]:


train_json['period'].value_counts()  # 统计train_json中'period'标签各类样本的数量


# In[ ]:


train_json['weather'].value_counts()   # 统计train_json中'whether'标签各类样本的数量


# ## 自定义数据集

# In[ ]:


# 样本图片信息转化，创建量化数据集
class WeatherDataset(Dataset):
    def __init__(self, df):
        super(WeatherDataset, self).__init__()
        self.df = df
        # 对样本图片的量化转换
        self.transform = T.Compose([
            T.Resize(size=(340,340)),
            T.RandomCrop(size=(256, 256)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])
    # 返回一个样本数据，当使用obj[index]时实际就是在调用obj.__getitem__(index)
    def __getitem__(self, index):
        file_name = self.df['filename'].iloc[index]
        img = Image.open(file_name)
        img = self.transform(img)
        return img,                paddle.to_tensor(self.df['period'].iloc[index]),                paddle.to_tensor(self.df['weather'].iloc[index])
    # 返回样本的数量，当使用len(obj)时实际就是在调用obj.__len__()
    def __len__(self):
        return len(self.df)


# In[ ]:


# loc表示location的意思；iloc中的loc意思相同，前面的i表示integer，所以它只接受整数作为参数。
train_dataset = WeatherDataset(train_json.iloc[:])  # 测试集与训练集相互独立，充分利用所有训练数据
print(len(train_dataset))
val_dataset = WeatherDataset(train_json.iloc[-600:])  # 取一部分训练集显示训练效果
print(len(val_dataset))
train_dataset1 = WeatherDataset(train_json.iloc[:2600]) # 取另一部分数据集做数据交替训练，增强训练效果
print(len(train_dataset1))
train_loader1 = DataLoader(train_dataset, batch_size=100, shuffle=True)  # 数据分批操作，设计一次喂进的数据量，并将数据打乱
print(len(train_loader1))
train_loader2 = DataLoader(train_dataset1, batch_size=100, shuffle=True)
print(len(train_loader2))
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)


# # 搭建模型

# In[ ]:


from paddle.vision.models import resnet18
# 构建预测模型，直接使用resnet18网络模型
class WeatherModel(paddle.nn.Layer):
    def __init__(self):
        super(WeatherModel, self).__init__()
        backbone = resnet18(pretrained=True) # 加载resnet18预训练模型对象
        backbone.fc = paddle.nn.Identity()  # 模型FC层
        self.backbone = backbone
        self.fc1 = paddle.nn.Linear(512, 4) # 时段标签Linear层
        self.fc2 = paddle.nn.Linear(512, 3) # 天气标签Linear层
    # 前向输出，返回FC层数据
    def forward(self, x):
        out = self.backbone(x)
        logits1 = self.fc1(out)
        logits2 = self.fc2(out)
        return logits1, logits2


# In[ ]:


model = WeatherModel() # 创建模型对象
model(paddle.to_tensor(np.random.rand(10, 3, 256, 256).astype(np.float32))) # 模型返回的FC层tensor量化


# # 训练与验证

# In[12]:


optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0000009) # 梯度滑动平均，偏差纠正优化
criterion = paddle.nn.CrossEntropyLoss() # 交叉熵计算损失
# 训练模块
for epoch in range(1, 201):
    # 双通道数据交替训练
    if epoch%2==0:
        train_loader = train_loader1
    else:
        train_loader = train_loader2
    print(f"第{epoch}轮：")
    Train_Loss, Val_Loss = [], []
    Train_ACC1, Train_ACC2 = [], []
    # 进入训练模式
    model.train()
    for i, (x, y1, y2) in enumerate(train_loader):
        pred1, pred2 = model(x) # 得到时段与天气的标签预测值
        loss = criterion(pred1, y1) + criterion(pred2, y2) # 计算预测值与真实值误差
        Train_Loss.append(loss.item()) # 记录每次训练的误差值
        loss.backward() # 根据loss来计算网络参数的梯度
        optimizer.step() # 针对计算得到的参数梯度对网络参数进行更新
        optimizer.clear_grad() # 清除梯度数据
        # 记录每批次数据的训练结果
        Train_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        Train_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())
        # 观测训练效果
        if (i+1)%10 == 0:
            print(f'\nEpoch: {i+1}')
            print(f'Loss: {loss.item():3.5f}')
            print(f'train_acc1：{(pred1.argmax(1) == y1.flatten()).numpy().mean():3.5f}')
            print(f'train_acc2：{(pred2.argmax(1) == y2.flatten()).numpy().mean():3.5f}')
# 测试训练的模型在训练集的表现
Val_ACC1, Val_ACC2 = [], []
model.eval()
for i, (x, y1, y2) in enumerate(val_loader):
    pred1, pred2 = model(x)
    loss = criterion(pred1, y1) + criterion(pred2, y2)
    Val_Loss.append(loss.item())
    Val_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
    Val_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())

# 综合训练效果展示
print('FIANLL:')
print(f'Loss {np.mean(Train_Loss):3.5f}/{np.mean(Val_Loss):3.5f}')
print(f'period.ACC {np.mean(Train_ACC1):3.5f}/{np.mean(Val_ACC1):3.5f}')
print(f'weather.ACC {np.mean(Train_ACC2):3.5f}/{np.mean(Val_ACC2):3.5f}')


# # 预测与提交

# In[ ]:


# 测试数据获取
import glob
test_df = pd.DataFrame({'filename': glob.glob('./test_images/*.jpg')})
test_df['period'] = 0
test_df['weather'] = 0
test_df = test_df.sort_values(by='filename')


# In[ ]:


test_dataset = WeatherDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# In[ ]:


# 模型预测模式
model.eval()
period_pred = []
weather_pred = []
# 遍历测试集，得到模型对测试集的预测结果
for i, (x, y1, y2) in enumerate(test_loader):
    pred1, pred2 = model(x)
    period_pred += period_dict[pred1.argmax(1).numpy()].tolist()
    weather_pred += weather_dict[pred2.argmax(1).numpy()].tolist()
# 取得预测结果
test_df['period'] = period_pred
test_df['weather'] = weather_pred


# In[ ]:


# 创建submit_json文件
submit_json = {
    'annotations':[]
}
# 将预测数据写入submit_json文件用于结果提交
for row in test_df.iterrows():
    submit_json['annotations'].append({
        'filename': 'test_images\\' + row[1].filename.split('/')[-1],
        'period': row[1].period,
        'weather': row[1].weather,
    })

with open('submit.json', 'w') as up:
    json.dump(submit_json, up)


# # 总结和展望
# 
# 1. 项目使用预训练模型 + 多分类分支完成天气分类和时间分类操作。
# 2. 可以考虑在项目中加入额外的数据扩增方法和交叉验证来改建模型精度。

# In[ ]:




