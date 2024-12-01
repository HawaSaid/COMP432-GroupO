# COMP432-Project

# Members of Group O <br/>

<u> Hawa-Afnane Said - 40263400</u>  <br/>
<u> Lesley Ventura - 40281652</u>  <br/>
<u> Marko Matijevic - 40282591</u>  <br/>
<u> Haifaa Janoudi - 40263748</u>  <br/>
<u> Sopheaktra Lean  - 40225014</u>  <br/>

# Description of the Project <br/>
<b>This project focuses on applying Convolutional Neural Networks (CNNs) to image classification tasks across three diverse datasets:</b>

1. Colorectal Cancer Classification: Medical imaging dataset to classify cancerous and non-cancerous samples.
2. Prostate Cancer Classification: Another medical imaging dataset, testing the generalizability of models trained on a similar medical domain.
3. Animal Faces Classification: A non-medical dataset, used to evaluate the domain adaptability of pre-trained and custom-trained CNN models.

<b>Key objectives:</b><br/>

- Train a CNN model (ResNet-50) on the Colorectal Cancer dataset, leveraging state-of-the-art techniques like transfer learning, feature extraction, and hyperparameter tuning.
- Explore feature transfer and extraction capabilities across different domains
- Compare the performance of a domain-specific encoder (trained on Colorectal Cancer) with a general-purpose ImageNet-pretrained encoder using t-SNE visualization
- Apply classical machine learning techniques for feature classification like Random Forests 

<b>Methodology Highlights:</b>

- Model Design and Training: The project used ResNet-50 for custom training on the Colorectal Cancer Dataset. The training involved techniques like batch normalization, dropout layers, and optimization via the Adam optimizer.
- Feature Analysis: Features extracted by both the Task 1-trained encoder and the ImageNet-pretrained encoder were visualized using t-SNE to understand their representational capabilities.
- Transfer Learning: The project explored the adaptability of pre-trained CNNs in extracting features from datasets in different domains.
- Performance Evaluation: Precision, recall, F1 score, and accuracy metrics were used to evaluate the classification performance on the prostate cancer and animal faces datasets.

<b>Key Contributions:</b>

- Demonstrated how a domain-specific model (Task 1-trained encoder) performs well in a similar medical domain but struggles with cross-domain datasets.
- Showcased the robustness of the ImageNet-pretrained encoder in handling diverse datasets.
- Highlighted the utility of visual tools like t-SNE in explaining feature separability and class distinctions.

# How to run the Python Code <br/>

1. Install Pytorch
   - For CPU only:<br/>
     `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` <br/>
   - For GPU using CUDA:<br/>
     `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` <br/>
2. Install all the required modules
   For scikit-learn:<br/>
   `pip install scikit-learn` <br/>
    For numpy:<br/>
   `pip install numpy`<br/>
    For shutil: <br/>
   `pip install shutil`<br/>
   For Matplotlib: <br/>
   `pip install matplotlib`<br/>
   For os: <br/>
   `pip install os`<br/>

# Source Code Package in PyTorch
src/
  - Dataset 1
  - Dataset 2
  - Dataset 3
  - README.md
  - task2-ImageNet.py
  - task2-PreTrainedModel.py
  - test.py

# Obtain the Datasets <br/>
<ol>
  <li>Dataset 1: Colorectal Cancer Classification [<a href="https://zenodo.org/records/1214456">Original Dataset</a> | <a href="https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp">Project Dataset</a>]</li>
  <li>Dataset 2: Prostate Cancer Classification [<a href="https://zenodo.org/records/4789576">Original Dataset</a> | <a href="https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp">Project Dataset</a>]</li>
  <li>Dataset 3: Animal Faces Classification [<a href="https://www.kaggle.com/datasets/andrewmvd/animal-faces">Original Dataset
  </a> | <a href="https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp">Project Dataset</a>]</li>
</ol>

