import torch
import torch.nn as nn
import time

def train(model, train_dl, criterion, optimizer, device, epoch):
    model.train()
    train_loss, correct, total = 0, 0, 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # For feature map augmentation, we need to handle the doubled batch size
        batch_size = inputs.size(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Duplicate targets for augmented features
        targets_augmented = torch.cat((targets, targets), dim=0)
        loss = criterion(outputs, targets_augmented)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets_augmented.size(0)
        correct += predicted.eq(targets_augmented).sum().item()

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: TrainLoss {train_loss/(batch_idx+1):.3f} | TrainAcc {acc:.2f}% | Time {time.time()-start_time:.2f}s")
    return acc

def validate(model, val_dl, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc