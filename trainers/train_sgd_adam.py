# trainers/train_gd.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_gd(
    model,
    train_loader,
    criterion,
    optimizer_type="SGD",
    lr=0.01,
    num_epochs=500,
    plot_convergence=True
):
    # Choose optimizer
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    device = next(model.parameters()).device
    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_history.append(epoch_loss / len(train_loader))

        # Evaluate accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total
        accuracy_history.append(accuracy)

        if epoch % 50 == 0:
            print(f"[{optimizer_type}] Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}")

    if plot_convergence:
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{optimizer_type} Training Convergence')
        plt.show()

    return model, loss_history, accuracy_history
