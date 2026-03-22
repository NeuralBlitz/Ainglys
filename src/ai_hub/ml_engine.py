import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from acca import CoreModule


class MLEngine:
    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 10
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CoreModule(input_dim, hidden_dim, output_dim).to(self.device)
        self.is_trained = False

    def train(
        self, X: torch.Tensor, y: torch.Tensor, epochs: int = 10
    ) -> Dict[str, float]:
        X, y = X.to(self.device), y.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.is_trained = True
        return {"final_loss": losses[-1], "epochs": epochs}

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.predict(X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y.to(self.device)).sum().item() / y.size(0)
        return {"accuracy": accuracy}

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_trained = True
