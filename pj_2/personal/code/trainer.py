import sched
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss, SmoothL1Loss, BCEWithLogitsLoss

import torch.optim as optim
import time
from time import sleep
from tqdm import tqdm
import os
from pathlib import Path


## 模型权重初始化
def initial_para(net=None):
    assert net!=None,'权重初始化没有传入网络'
    for module in net.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)
            module.eps = 0.00001
            module.momentum = 0.1
        else:
            module.float()
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
    torch.cuda.empty_cache()
    return net


def train(device, model, train_loader, test_loader, num_classes, weight_decay=5e-4, lr=1e-2, scheduler=None, max_epoch=50, loss='CrossEntropyLoss'):
    big1 = 0
    big2 = 0
    stage2 = False
    trainACCes = []
    testACCes = []
    losses = []
    net = model.to(device)
    net = initial_para(net)
    criterion = globals()[loss]()
    optimizer = optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    if scheduler == "OneCycleLR":
        scheduler0 = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.4,
            three_phase=True,
            steps_per_epoch=len(train_loader),
            epochs=max_epoch,
        )
    if scheduler == "CyclicLR":
        scheduler0 = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, step_size_up=2, step_size_down=16, max_lr=0.2)
    if scheduler == "Cos":
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)
        scheduler0 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    loop = tqdm(
        range(max_epoch),
        unit="epoch",
        bar_format="{n_fmt}/{total_fmt}|{bar}| [{rate_fmt}{postfix}]\t",
    )
    for epoch in loop:
        net.train()
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = F.one_hot(labels, num_classes=num_classes).float()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            del inputs, labels
            if scheduler in ["OneCycleLR"]:
                scheduler0.step()
        if scheduler in ["CyclicLR", "Cos"]:
            scheduler0.step()
        if (
            scheduler == "CyclicLR"
            and stage2 == False
            and scheduler0.get_last_lr()[0] <= lr
        ):
            stage2 = True
            scheduler0 = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1 - 1e-2)

        trainACC, testACC = eval(device, net, train_loader, test_loader)
        losses.append(loss.item())
        trainACCes.append(trainACC)
        testACCes.append(testACC)
        big1 = max(big1, trainACC)
        big2 = max(big2, testACC)

        if scheduler is not None:
            loop.set_postfix_str(
                "loss:%.2f, lr:%.5f, trainacc:%.2f, testacc:%.2f"
                % (
                    running_loss / len(train_loader),
                    scheduler0.get_last_lr()[0],
                    trainACC,
                    testACC,
                )
            )
        else:
            loop.set_postfix_str(
                "loss:%.2f, lr:%.4f, trainacc=%.2f, testacc=%.2f"
                % (
                    running_loss / len(train_loader),
                    optimizer.state_dict()["param_groups"][0]["lr"],
                    trainACC,
                    testACC,
                )
            )

    return losses, trainACCes, testACCes


def eval(device, net, trainset=None, testset=None, oncuda=True, SetType='test'):
    net.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        if trainset != None:
            for data in trainset:
                images, labels = data
                if oncuda:
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            trainacc = 100 * correct / total
        total = 0
        correct = 0
        if testset != None:
            for data in testset:
                images, labels = data
                if oncuda:
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            testacc = 100 * correct / total
    torch.cuda.empty_cache()
    return trainacc, testacc


def test(device, net, train_loader, test_loader):
    trainACC, testACC = eval(device, net, train_loader, test_loader)
    print("Accuracy of the network on train set: %.3f %%" % (trainACC))
    print("Accuracy of the network on test set: %.3f %%" % (testACC))
