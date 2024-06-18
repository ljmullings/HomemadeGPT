import torch
import torch.nn as nn
import torch.optim as optim
from gpt import GPT
import os

# hyperparameters
embed_size = 512
heads = 8
forward_expansion = 4
num_layers = 6
vocab_size = 50257  
max_len = 25
dropout = 0.1
learning_rate = 1e-4  # stability
batch_size = 32
total_epochs = 12  # total number of epochs you want to train for in total

# initialize the model
model = GPT(embed_size, heads, forward_expansion, num_layers, vocab_size, max_len, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# path to save/load the model
model_path = 'gpt_model.pth'

# training from scratch if a model doesn't exist
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load saved model state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load saved optimizer state
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {start_epoch} with loss {loss:.4f}")
else:
    start_epoch = 0
    print("Starting training from scratch")

# dummy data, will be replaced with real data soon
if not os.path.exists('data.pth') or not os.path.exists('target.pth'):
    data = torch.randint(0, vocab_size, (batch_size, max_len))
    target = torch.randint(0, vocab_size, (batch_size, max_len))
    torch.save(data, 'data.pth')
    torch.save(target, 'target.pth')
    print("Generated and saved new dummy data.")
else:
    data = torch.load('data.pth')
    target = torch.load('target.pth')
    print("Loaded existing dummy data.")

# training loop
for epoch in range(start_epoch, total_epochs):
    model.train()
    optimizer.zero_grad()
    mask = None  # Not sure yet what to do with my mask
    outputs = model(data, mask)
    loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}')

    # save the model and optimizer state
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, model_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

print("Training complete.")
