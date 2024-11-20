import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

# set device
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA for NVIDIA GPUs
    print("\nUsing CUDA (GPU).\n")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS for Apple Silicon GPUs
    print("\nUsing MPS (Apple Silicon GPU).\n")
else:
    device = torch.device("cpu")  # Fall back to CPU
    print("\nUsing CPU.\n")

# load pre-trained ImageNet encoder model
imagenet_encoder = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)     ## try ResNet-34 and ResNet-101, or a completely different encoder ##
imagenet_encoder = torch.nn.Sequential(*list(imagenet_encoder.children())[:-1])
imagenet_encoder.to(device)

# hyperparameters
img_size = (224, 224)
batch_size = 64     ## try different batch size ##

# image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):    # fixes error about the image extension
                        self.image_paths.append(img_path)
                        self.labels.append(class_label)
                    # self.image_paths.append(os.path.join(class_dir, img_name))
                    # self.labels.append(class_label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    

# datasets paths
dataset2_path = 'COMP432-GroupO/Dataset 2/Prostate Cancer'
dataset3_path = 'COMP432-GroupO/Dataset 3/Animal Faces'

# dataloaders
dataloader2 = DataLoader(CustomImageDataset(dataset2_path, transform=transform), batch_size=batch_size, shuffle=False)
dataloader3 = DataLoader(CustomImageDataset(dataset3_path, transform=transform), batch_size=batch_size, shuffle=False)

def extract_features(encoder, dataloader):
    encoder.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device).float()
            output = encoder(images).squeeze()
            features.append(output.cpu().numpy())
            labels.extend(label.numpy())
    
    return np.vstack(features), np.array(labels)

# extract features for the ImageNet encoder on Dataset 2 and Dataset 3
features_imagenet_ds2, labels_ds2 = extract_features(imagenet_encoder, dataloader2)
features_imagenet_ds3, labels_ds3 = extract_features(imagenet_encoder, dataloader3)

def visualize_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.title(title)
    plt.show()

# visualize t-SNE results for Dataset 2 and Dataset 3 using ImageNet encoder features
visualize_tsne(features_imagenet_ds2, labels_ds2, "ImageNet Encoder on Dataset 2")
visualize_tsne(features_imagenet_ds3, labels_ds3, "ImageNet Encoder on Dataset 3")

## could also perform random forest, or logistic regression rather than SVM ##

# # perform classification on extracted features (SVM)
# clf = SVC()
# clf.fit(features_imagenet_ds2, labels_ds2)  # train on extracted features from Dataset 2
# accuracy2 = clf.score(features_imagenet_ds2, labels_ds2)
# print(f"Classification accuracy on Dataset 2: {accuracy2}")

# # perform classification on extracted features (SVM)
# clf.fit(features_imagenet_ds3, labels_ds3)  # train on extracted features from Dataset 3
# accuracy3 = clf.score(features_imagenet_ds3, labels_ds3)
# print(f"Classification accuracy on Dataset 3: {accuracy3}")

# perfrom classification on Dataset 2 with Random Forest
clf_rf_ds2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf_ds2.fit(features_imagenet_ds2, labels_ds2)
accuracy_rf_ds2 = clf_rf_ds2.score(features_imagenet_ds2, labels_ds2)
print(f"Random Forest classification accuracy on Dataset 2: {accuracy_rf_ds2}")

clf_rf_ds3 = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf_ds3.fit(features_imagenet_ds3, labels_ds3)
accuracy_rf_ds3 = clf_rf_ds3.score(features_imagenet_ds3, labels_ds3)
print(f"Random Forest classification accuracy on Dataset 3: {accuracy_rf_ds3}")

