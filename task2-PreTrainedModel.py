import torch
import torch.nn as nn
from torchvision import datasets, transforms, models  
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_value
import matplotlib.pyplot as plot
import numpy as np

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA for NVIDIA GPUs
    print("\nUsing CUDA (GPU).\n")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS for Apple Silicon GPUs
    print("\nUsing MPS (Apple Silicon GPU).\n")
else:
    device = torch.device("cpu")  # Fall back to CPU if neither are being used
    print("\nUsing CPU.\n")

# Transformation for datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)), #Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
])

# Get the path of dataset2 and dataset3
dataset2_path = './Dataset 2/Prostate Cancer'
dataset3_path = './Dataset 3/Animal Faces'

# Load datasets and dataloaders
dataset2 = datasets.ImageFolder(dataset2_path, transform=transform)
dataset3 = datasets.ImageFolder(dataset3_path, transform=transform)

dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=False)
dataloader3 = torch.utils.data.DataLoader(dataset3, batch_size=32, shuffle=False)

# Load the pretrained model from task1
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

# Initialize the model wights and load them 
model = CustomResNet(num_classes=3).to(device)
model.load_state_dict(torch.load('my_model_task1_lr0.001_fulldataset.pth', map_location=device, weights_only=True))

# Remove classification head (final layer)
encoder = nn.Sequential(*list(model.children())[:-1])  # Removing the final FC layer
encoder = encoder.to(device)
encoder.eval()

# Function to extract features
def extract_features(dataloader, encoder, device):
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = encoder(inputs)
            features.append(outputs.view(outputs.size(0), -1).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Extract features from Dataset 2 and Dataset 3
features_CNN_dataset2, label_ds2 = extract_features(dataloader2, encoder, device)
features_CNN_ds3, label_ds3 = extract_features(dataloader3, encoder, device)

# t-SNE visualization
def visualize_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plot.figure(figsize=(10, 6))
    plot.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plot.colorbar()
    plot.title(title)
    plot.show()

# Visualize t-SNE results for Dataset 2 and Dataset 3 using the PreTrained Model
visualize_tsne(features_CNN_dataset2, label_ds2, "PreTrained Model on Dataset 2")
visualize_tsne(features_CNN_ds3, label_ds3, "PreTrained Model on Dataset 3")

# Train-test split for classification
def train_classifying(features, labels, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_prediction = classifier.predict(X_test)
    print(f"\nClassification Report for {dataset_name}:")
    print(classification_report(y_test, y_prediction))
    print(f"Accuracy for {dataset_name}: {accuracy_value(y_test, y_prediction):.4f}")

# Apply classification on features extracted from Dataset 2 and Dataset 3

train_classifying(features_CNN_dataset2, label_ds2, "Dataset 2")
train_classifying(features_CNN_ds3, label_ds3, "Dataset 3")