import torch
import torch.nn as nn
import numpy as np

class SimonMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 5 צבעים אחרונים + 2 זמני תגובה + רמת סיבוב = 8 פיצ'רים
input_dim = 8
output_dim = 4  # ארבעה צבעים

# דאטה פיקטיבי (לצורך הדגמה) – תחליפי בנתונים אמיתיים כשתרצי
N = 2000
X = np.random.randint(0, 4, size=(N, 5))    # רצף צבעים
R = np.random.rand(N, 2)                    # זמני תגובה
S = np.random.rand(N, 1)                    # רמת סיבוב
X_all = np.concatenate([X, R, S], axis=1)
y = np.random.randint(0, 4, size=(N,))      # הצבע הבא (label)

X_t = torch.tensor(X_all, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

model = SimonMLP(input_dim, output_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(300):
    optimizer.zero_grad()
    out = model(X_t)
    loss = loss_fn(out, y_t)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss {loss.item():.3f}")

# שמירה
torch.save(model.state_dict(), "../../../../../../../Desktop/robotics/roboFriend2506/simon/simon_mlp.pt")
print("✅ Model saved as simon_mlp.pt")
