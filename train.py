import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import TransformerLanguageModel
from text_dataset import DiscordDataset

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0.0
    for input_seq, target_seq in dataloader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        seq_len = input_seq.size(1)
        src_mask = generate_square_subsequent_mask(seq_len).to(device)
        output = model(input_seq, src_mask)

        # Reshape output and target for computing loss
        output_flat = output.view(-1, output.size(-1))
        target_flat = target_seq.view(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        
        # Gradient clipping (best practice for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            seq_len = input_seq.size(1)
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            output = model(input_seq, src_mask)

            output_flat = output.view(-1, output.size(-1))
            target_flat = target_seq.view(-1)
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Hyperparameters and device setup
    batch_size = 32
    epochs = 10
    lr = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NOTE: Adjust vocab_size based on your dataset vocabulary.
    vocab_size = 10000  
    model = TransformerLanguageModel(vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming <pad> token index is 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Optionally, use a learning rate scheduler (e.g., step scheduler or Noam scheduler).
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
    # Load the custom dataset (ensure discord_messages.csv is available)
    dataset = DiscordDataset(csv_file="discord_messages.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, optimizer, criterion, dataloader, device)
        print(f"Epoch {epoch} | Training Loss: {train_loss:.4f}")
        scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), "transformer_language_model.pth")

if __name__ == "__main__":
    main()
