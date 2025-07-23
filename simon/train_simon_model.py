import torch
import torch.nn as nn
import numpy as np
import pickle

class SimonAI_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.color_out = nn.Linear(32, 12)
        self.len_out = nn.Linear(32, 3)
        self.scale_out = nn.Linear(32, 4)
        self.comp_out = nn.Linear(32, 1)
        self.tempo_out = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        colors = self.color_out(h)
        step_len = self.len_out(h)
        scale = self.scale_out(h)
        comp = self.comp_out(h)
        tempo = self.tempo_out(h)
        return colors, step_len, scale, comp, tempo

input_dim = 8


with open("simon_data.pkl", "rb") as f:
    collected_data = pickle.load(f)


X = np.array([d['features'] for d in collected_data])
y_colors = np.array([d['y_colors'] for d in collected_data])
y_len = np.array([d['y_len'] for d in collected_data])
y_scale = np.array([d['y_scale'] for d in collected_data])
y_comp = np.array([d['y_comp'] for d in collected_data])
y_tempo = np.array([d['y_tempo'] for d in collected_data])


# === המרה ל־Torch Tensors ===
X_t = torch.tensor(X, dtype=torch.float32)
y_colors_t = torch.tensor(y_colors, dtype=torch.long)
y_len_t = torch.tensor(y_len, dtype=torch.long)
y_scale_t = torch.tensor(y_scale, dtype=torch.long)
y_comp_t = torch.tensor(y_comp, dtype=torch.float32).unsqueeze(1)
y_tempo_t = torch.tensor(y_tempo, dtype=torch.float32).unsqueeze(1)

model = SimonAI_MLP(input_dim)
loss_fn_colors = nn.CrossEntropyLoss()
loss_fn_len = nn.CrossEntropyLoss()
loss_fn_scale = nn.CrossEntropyLoss()
loss_fn_comp = nn.MSELoss()
loss_fn_tempo = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(300):
    optimizer.zero_grad()
    colors_pred, len_pred, scale_pred, comp_pred, tempo_pred = model(X_t)

    # עיצוב מחדש של הפלטים
    colors_pred = colors_pred.view(-1, 3, 4)  # (N, 3, 4)
    loss_colors = sum(
        loss_fn_colors(colors_pred[:, i, :], y_colors_t[:, i])
        for i in range(3)
    ) / 3

    loss_len = loss_fn_len(len_pred, y_len_t)
    loss_scale = loss_fn_scale(scale_pred, y_scale_t)
    loss_comp = loss_fn_comp(comp_pred, y_comp_t)
    loss_tempo = loss_fn_tempo(tempo_pred, y_tempo_t)

    loss = loss_colors + loss_len + loss_scale + loss_comp + loss_tempo
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss {loss.item():.3f}")

torch.save(model.state_dict(), "simon_mlp.pt")
print("✅ Model saved as simon_mlp.pt")
