import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Split data into test and train, evenly distributing the targets
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)

# Standardise the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for training and test sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle so order is different each time
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class IrisClassificationNet(nn.Module):
    def __init__(self, hidden_units=8):
        super(IrisClassificationNet, self).__init__()
        self.fc1 = nn.Linear(4, hidden_units)  # 4 input features
        self.fc2 = nn.Linear(hidden_units, 3)  # 3 output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply relu function to introduce non-linearity
        x = self.fc2(x)
        return x


model = IrisClassificationNet(hidden_units=8)

criterion = nn.CrossEntropyLoss()  # Loss function
optimiser = optim.Adam(model.parameters(), lr=0.01)

epochs = 20
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimiser.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation phase on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():  # Do not compute gradients as not required for eval
        for X_batch, y_batch in test_loader:
            test_outputs = model(X_batch)
            loss = criterion(test_outputs, y_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

# Calculate accuracy
predictions = torch.argmax(model(X_train), dim=1)
num_correct = (predictions == y_train).sum().item()
train_accuracy = num_correct / len(X_train)

predictions = torch.argmax(model(X_test), dim=1)
num_correct = (predictions == y_test).sum().item()
test_accuracy = num_correct / len(X_test)
print(f'Train Accuracy = {train_accuracy}%, Test Accuracy = {test_accuracy}%')


# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
