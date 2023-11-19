# импорт
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

trans = tv.transforms.Compose([tv.transforms.ToTensor()])

# датасет
ds_mnist = tv.datasets.MNIST('./datasets', download=True, transform=trans)
#print(ds_mnist[0])
#print(ds_mnist[0][0].shape)
#plt.imshow(ds_mnist[0][0].numpy()[0])
#plt.show()

# загрузчик, dataloader
batch_size = 16
dataloader = torch.utils.data.DataLoader(ds_mnist, batch_size=batch_size, shuffle=True, drop_last=True)

#for img, label in dataloader:
 #print(img.shape)
 #print("label =")
 #print(label)
 #print(label.shape)
 #break

 # нейронная сеть
class Neural_numbers(nn.Module):
     def __init__(self):
         super().__init__()
         self.flat = nn.Flatten()
         # 2-а полносвязных слоя
         self.linear1 = nn.Linear(28*28, 100)
         self.linear2 = nn.Linear(100, 10)

         # функция активации
         self.act = nn.ReLU()

     def forward(self, x):
         out = self.flat(x)
         out = self.linear1(out)
         out = self.act(out)
         out = self.linear2(out)

         return out

# фукция вычисления параметров сети
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

     # функция потерь

# сеть, объект класса
model = Neural_numbers  ()
model_1 = Neural_numbers()
model_2 = Neural_numbers()
model_3 = Neural_numbers()
model_4 = Neural_numbers()
model_5 = Neural_numbers()
model_6 = Neural_numbers()
model_7 = Neural_numbers()
model_8 = Neural_numbers()
model_9 = Neural_numbers()
list_nst = ['model', 'model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9']
correct_percentage = []
# вычисляем параметры
print(count_parameters(model))

# функция потерь
loss_fn = nn.CrossEntropyLoss()

# оптимизатор
optimezer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)

def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    """print("answer.shape =")
    print(answer.shape)
    print("answer =")
    print( answer)
    print("answer.mean =")
    print(answer.mean())"""
    return answer.mean()


epochs = 10
for epoch in range(epochs):
    loss_val = 0
    acc_val = 0

    # цикл обучения
    for img, label in (pbar:= tqdm(dataloader)):
        # обнуление градиентов
        optimezer.zero_grad()

        # преобразуем метку изображения цифры к виду 00001000 по количеству классов
        label = F.one_hot(label, 10).float()
        # получаем предсказание
        pred = model(img)
        # получаем функцию потерь
        loss = loss_fn(pred,label)
        loss.backward()
        loss_item = loss.item()
        loss_val += loss_item

        optimezer.step()

        acc_current = accuracy(pred, label)
        acc_val += acc_current

        pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')

    correct_percentage.append(loss_val/len(dataloader))

accuracy(pred, label)
for i in range(9):
    print(list_nst[i],"процент правильных выборов", (correct_percentage[i]) * 100)
