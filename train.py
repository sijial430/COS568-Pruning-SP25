import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import time

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10, use_amp=False):
    model.train()

    scaler = GradScaler(enabled=use_amp)  # Only scales if use_amp=True
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        #output = model(data)
        #train_loss = loss(output, target)
        #with autocast(enabled=use_amp):
        with autocast(device_type='cuda', enabled=use_amp):
            output = model(data)
            train_loss = loss(output, target)
        
        total += train_loss.item() * data.size(0)
        #train_loss.backward()
        #optimizer.step()
        
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        
        
        
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, use_amp=False):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5, np.nan]]
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose, use_amp=use_amp)
        start_time = time.time()
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        end_time = time.time()
        row = [train_loss, test_loss, accuracy1, accuracy5, end_time - start_time]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy', 'testing_time']
    return pd.DataFrame(rows, columns=columns)


