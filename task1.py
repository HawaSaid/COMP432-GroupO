import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
import shutil

# GPU set up
print(torch.__version__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and set for use.")
else:
   device = torch.device("cpu")
   print("No GPU available. Using CPU.")

# naming datasets/paths - original dataset used for submission-wise work, limited_dataset, can put 50 files if i want, used to run code
# to check if it will run to completion.

# -- Original Dataset Path (comment the next 2 lines out to use the sample dataset) -- #
original_dataset = 'Dataset 1/Colorectal Cancer'
limited_dataset = 'path/'  # temp directory for the limited dataset

# -- Sample Dataset Path (uncomment the next 2 lines to use the sample dataset) -- #
# original_dataset = 'Sample Datasets/Sample Dataset 1/Colorectal Cancer'
# limited_dataset = 'path/'  # temp directory for the limited dataset

# Limited dataset created here, max_files_per_class = x, where x amount of files will be used per class
# Function to create a limited dataset
def create_limited_dataset(original_dir, limited_dir, max_files_per_class=50):
    if os.path.exists(limited_dir):
        shutil.rmtree(limited_dir)
    os.makedirs(limited_dir)

    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if os.path.isdir(class_dir):
            limited_class_dir = os.path.join(limited_dir, class_name)
            os.makedirs(limited_class_dir)

            files = os.listdir(class_dir)[:max_files_per_class]
            for file_name in files:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(limited_class_dir, file_name)
                shutil.copy(src, dst)


# ---------------------------------------------------------------------------------------------------------
#                                          Hyper-parameters
# ---------------------------------------------------------------------------------------------------------
img_size = (224, 224)
batch_size = 128
num_classes = 3
epochs = 40
learning_rate = 0.001

# image generator for loading image data and augmenting images, done to make training better, less stagnant behaviour
# should output better results for when we run testing in task 2
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# change original_dataset to limited_dataset for using the limited dataset to train - [debugging]
train_dataset = ImageFolder(original_dataset, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------------------
#                                                                                      |
#                                     CNN Management                                   |
#                                                                                      |
# ---------------------------------------------------------------------------------------

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.base_model = models.resnet50(weights=True)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


#Initialize the model and move it to the appropriate device
model = CustomResNet(num_classes=num_classes).to(device)

#Freeze base layers in model
for param in model.base_model.parameters():
    param.requires_grad = False  # Freezes only the base layers

#Ensure the final fully connected layers are trainable
for param in model.base_model.fc.parameters():
    param.requires_grad = True

#optimizer and loss function after updating requires_grad settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

epoch_losses = []  #store loss for each epoch
# train using training generator
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# ---------------------------------------------------------------------------------------------------------
#                                          Plotting
# ---------------------------------------------------------------------------------------------------------
# plot accuracy (in this PyTorch version, only loss is tracked for simplicity)
#print(epoch_losses)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# extract features using CNN encoder
model.eval()
features = []
labels_list = []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        feature_batch = model.base_model(inputs)
        features.append(feature_batch.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

features = np.concatenate(features)
labels_list = np.concatenate(labels_list)

# get t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features)

# plot t-SNE results
plt.figure(figsize=(8, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_list, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE Visualization of CNN Features')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# saving the encoder model to a file... Done using torch.save for the model weights
torch.save(model.state_dict(), 'my_model_task1_lr0.001_fulldataset.pth')
