import torch
import torch.nn as nn
import torch.optim as optim
from gpt import GPT
import os
import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

def main():
    # Hyperparameters
    embed_size = 512
    heads = 8
    forward_expansion = 4
    num_layers = 6
    max_len = 100  
    dropout = 0.3 
    learning_rate = 5e-5 
    batch_size = 64  # Increased batch size
    total_epochs = 2

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Initialize the model
    model = GPT(embed_size, heads, forward_expansion, num_layers, vocab_size, max_len, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Added weight decay

    # Path to save/load the model
    model_path = './models/gpt_modelv5.pth'

    # Training from scratch if a model doesn't exist
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {loss:.4f}")
    else:
        start_epoch = 0
        print("Starting training from scratch")

    # Load tokenized data
    tokenized_data_path = './training/tokenized_data.pt'
    data, target = torch.load(tokenized_data_path)
    print(f"Tokenized data loaded from {tokenized_data_path}")

    # Ensure token indices are within the vocab_size range
    data = data.clamp(0, vocab_size - 1).to(device)
    target = target.clamp(0, vocab_size - 1).to(device)

    # Debug: Check data shapes
    print("Data shape:", data.shape)
    print("Target shape:", data.shape)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(data, target)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Debug: Check dataset sizes
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Added num_workers
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{total_epochs}") as pbar:
            for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
                optimizer.zero_grad()
                mask = None  # Adjust mask handling as needed
                outputs = model(batch_data, mask)
                loss = criterion(outputs.view(-1, vocab_size), batch_target.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        avg_train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{total_epochs}] complete in {epoch_time:.2f}s, Average Training Loss: {avg_train_loss:.4f}')

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_outputs = model(val_data, mask)
                loss = criterion(val_outputs.view(-1, vocab_size), val_target.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{total_epochs}], Validation Loss: {avg_val_loss:.4f}')

        # Save  model and optimizer state
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, model_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    print("Training complete.")

if __name__ == '__main__':
    main()
