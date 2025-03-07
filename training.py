# training.py

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from models import ImitationNet
import config

# Custom dataset to load expert demonstrations.
class DemoDataset(Dataset):
    def __init__(self, demos_path):
        with open(demos_path, 'rb') as f:
            demos = pickle.load(f)
        self.data = []
        # Flatten episodes into state and the first agent’s action.
        for episode in demos:
            for state, actions in episode:
                self.data.append((state, actions[0]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action = self.data[idx]
        # Convert state to tensor (assuming state is a flat vector or can be flattened)
        state_tensor = torch.tensor(state, dtype=torch.float32).view(-1)
        action_tensor = torch.tensor(action, dtype=torch.long)
        return state_tensor, action_tensor

def train_model():
    dataset = DemoDataset(config.DEMO_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Infer input dimension from a sample state.
    sample_state, _ = dataset[0]
    input_dim = sample_state.shape[0]
    output_dim = config.NUM_ACTIONS  # Number of discrete actions
    
    model = ImitationNet(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        epoch_loss = 0.0
        for states, actions in dataloader:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
