import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class SymptomDataset(Dataset):
    def __init__(self, texts, labels, encoder):
        self.embeddings = encoder.encode(texts, convert_to_tensor=True)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class MultiTaskHeads(nn.Module):
    def __init__(self, input_dim, output_dims):
        super().__init__()
        # Definimos cabezales separados para cada síntoma
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for name in output_dims
        })
        self.output_names = output_dims

    def forward(self, x):
        results = {}
        for name in self.output_names:
            results[name] = self.heads[name](x)
        return results

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Load data
    df = pd.read_csv("scripts/data/train_v1.csv")
    symptoms = ["pain", "fatigue", "anxiety", "mood", "sleep"]
    
    # Encoder
    model_name = "all-MiniLM-L6-v2"
    encoder = SentenceTransformer(model_name).to(device)
    
    dataset = SymptomDataset(df["text"].tolist(), df[symptoms], encoder)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Model
    model = MultiTaskHeads(input_dim=384, output_dims=symptoms).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Simple training loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_emb, batch_labels in loader:
            batch_emb, batch_labels = batch_emb.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_emb)
            
            loss = 0
            for i, name in enumerate(symptoms):
                loss += criterion(outputs[name], batch_labels[:, i].unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Export to ONNX
    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 384).to(device)
    
    # Necesitamos exportar el modelo PyTorch
    torch.save(model.state_dict(), "heads_v1.pth")
    
    # Export heads to ONNX
    # (Para el MVP, mantendremos el encoder de SentenceTransformer separado o usaremos optimum para el wrap completo)
    # Por ahora exportamos solo los cabezales
    output_names = symptoms
    torch.onnx.export(
        model,
        dummy_input,
        "heads_v1.onnx",
        input_names=["embeddings"],
        output_names=output_names,
        dynamic_axes={"embeddings": {0: "batch_size"}}
    )
    print("Optimization complete: heads_v1.onnx generated.")

if __name__ == "__main__":
    train()
