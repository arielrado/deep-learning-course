import torch
import statistics

class Model:
    def fit(self, train_loader, val_loader, epochs, lr, weight_decay):
        losses = []
        training_accuracy = []
        validation_accuracy = []
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            epoch_accuracy = []
            for images, labels in train_loader:
                images = images.requires_grad_().to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_accuracy.append(self.evaluate(images, labels))
            
            losses.append(loss.item())
            training_accuracy.append(sum(epoch_accuracy) / len(epoch_accuracy))
            validation_accuracy.append(self.evaluate(*next(iter(val_loader))))
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Training Accuracy: {training_accuracy[-1]}, Validation Accuracy: {validation_accuracy[-1]}')

        return losses, training_accuracy, validation_accuracy
     
    def predict(self, x):
        x = x.to(self.device)
        outputs = self(x)
        _, predicted = torch.max(outputs, 1)
        return predicted
    
    def evaluate(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        predicted = self.predict(x)
        correct = (predicted == y).sum().item()
        return correct / y.size(0)
    
    def batched_evaluate(self, data_loader : torch.utils.data.DataLoader):        
        return statistics.mean([self.evaluate(images, labels) for images, labels in data_loader])
    