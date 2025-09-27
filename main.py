import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import torchvision.transforms as transforms

from data import get_dataloaders
from model import ResNet, BasicBlock
from train_eval import train, validate

# Global variables for augmentation
t0 = transforms.Compose([])
best_total = 0

def main():
    root = "/kaggle/input/lrwrw/CLAHE_images"
    train_dl, val_dl, test_dl = get_dataloaders(root)

    # Augmentation hyperparameters
    H = [0, 0.5]  # Horizontal flip probability
    V = [0, 0.5]  # Vertical flip probability  
    R = [0, 15]   # Rotation degrees
    S = [0, 0.4]  # Scale factor
    combinations = list(itertools.product(H, V, R, S))
    df_combinations = pd.DataFrame(combinations, columns=['H','V','R','S'])
    print("Augmentation parameter combinations:")
    print(df_combinations)

    Y = []
    for i in range(len(combinations)):
        print(f"\n=== Testing combination {i+1}/{len(combinations)} ===")
        print(f"Parameters: H={df_combinations.iloc[i, 0]}, V={df_combinations.iloc[i, 1]}, "
              f"R={df_combinations.iloc[i, 2]}, S={df_combinations.iloc[i, 3]}")

        # Define transforms dynamically for feature map augmentation
        t1 = transforms.RandomHorizontalFlip(p=df_combinations.iloc[i, 0])
        t2 = transforms.RandomVerticalFlip(p=df_combinations.iloc[i, 1])
        t3 = transforms.RandomRotation(degrees=(0, df_combinations.iloc[i, 2]))
        t5 = transforms.RandomAffine(degrees=(0,0), scale=(1, 1+df_combinations.iloc[i, 3]))
        
        global t0
        t0 = transforms.Compose([t1, t2, t3, t5])

        # Initialize model
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        global best_total
        best_total = 0

        if device == 'cuda':
            model = torch.nn.DataParallel(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

        # Training and validation loop
        for epoch in range(50):
            train_acc = train(model, train_dl, criterion, optimizer, device, epoch)
            val_acc = validate(model, val_dl, criterion, device)
            
            if val_acc > best_total:
                best_total = val_acc

        Y.append(best_total)
        print(f"Best validation accuracy for this combination: {best_total:.2f}%")
        print(f"Current results: {Y}")

    # Statistical analysis of augmentation parameters
    df_combinations['correct'] = Y
    df_combinations['failures'] = 1000 - df_combinations['correct']  # Assuming 1000 validation samples
    y = df_combinations[['correct', 'failures']]
    X = df_combinations[['H','V','R','S']]
    X = sm.add_constant(X)

    # Generalized Linear Model analysis
    glm_model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    print("\n=== Statistical Analysis Results ===")
    print(glm_model.summary())
    
    # Odds Ratio calculation
    OR = np.exp(glm_model.params)
    print("\nOdds Ratios:")
    print(OR)

    # Save results
    results_df = df_combinations.copy()
    results_df['validation_accuracy'] = Y
    results_df.to_csv('augmentation_analysis_results.csv', index=False)
    print("\nResults saved to 'augmentation_analysis_results.csv'")

if __name__ == "__main__":
    main()