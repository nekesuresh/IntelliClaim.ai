import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

fraud_df = pd.read_csv("fraud_probabilities.csv")
damage_df = pd.read_csv("damage_probabilities.csv")

merged_df = fraud_df.merge(damage_df, left_on="claim_id", right_on="image_id", how="inner")

X = merged_df[["fraud_probability", "damage_probability"]].values
y = (merged_df["fraud_probability"] > 0.5).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class FusionNN(nn.Module):
    def __init__(self):
        super(FusionNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

model = FusionNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

with torch.no_grad():
    y_pred = model(X_test_torch).numpy().flatten()
    y_pred_class = (y_pred > 0.5).astype(int)

    print("Precision:", precision_score(y_test, y_pred_class))
    print("Recall:", recall_score(y_test, y_pred_class))
    print("F1-Score:", f1_score(y_test, y_pred_class))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))

fusion_results_df = pd.DataFrame({
    "claim_id": merged_df.iloc[y_test_torch.numpy().flatten().astype(bool)].claim_id.values,
    "final_fraud_probability": y_pred
})

fusion_results_df.to_csv("final_fraud_predictions.csv", index=False)
print("Final fraud predictions saved to 'final_fraud_predictions.csv'.")