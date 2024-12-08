# COMP432-Project

# Members of Group O <br/>

<u> Hawa-Afnane Said - 40263400</u>  <br/>
<u> Lesley Ventura - 40281652</u>  <br/>
<u> Marko Matijevic - 40282591</u>  <br/>
<u> Haifaa Janoudi - 40263748</u>  <br/>
<u> Sopheaktra Lean  - 40225014</u>  <br/>

# Links

- GitHub Repository: https://github.com/HawaSaid/COMP432-GroupO.git
- Presentation: 
- Distribution: https://docs.google.com/document/d/1nrpxcNu9BKhXDWJIeF3u13V0qrwCUe2aDYtKEMaZlPA/edit?usp=sharing

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

To run the Python code, 
1. Install PyTorch:
   - For CPU-only, use:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```
   - For GPU with CUDA, use:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
2. Install the required Python modules:
   ```bash
   pip install scikit-learn, numpy, matplotlib, pillow
   ```
   - ** Note that `os` and `shutil` are built-in modules, so no additional installation is needed.
3. Dataset Placement:
   - Download the 3 datasets with the links provided at the end. Ensure your datasets are placed in the specified directories
    ```bash
    Dataset 1/Colorectal Cancer
    Dataset 2/Prostate Cancer
    Dataset 3/Animal Faces
    ```
   - If you wish to use the sample datasets provided, follow these steps:
      - Uncomment the line specifying the path to the sample dataset in the code.
      - Comment out the line specifying the path to the original dataset in the code.
      - For example:
      ```bash
      # -- Original Datasets Paths (comment these out to use the sample datasets) -- #
      # dataset2_path = './Dataset 2/Prostate Cancer'
      # dataset3_path = './Dataset 3/Animal Faces'

      # -- Sample Dataset Path (uncomment this to use the sample dataset) -- #
      dataset2_path = './Sample Dataset 2/Prostate Cancer' 
      dataset3_path = './Sample Dataset 3/Animal Faces'
      ```
4. Execute the scripts using a terminal or IDE with the following command:
   - Task 1: Train and Validate our Model:
     ```bash
     python task1.py
     ```
   - Task 2 using Encoder from Task 1:
     ```bash
     python task2-PreTrainedModel.py
     ```
   - Task 2 using ImageNet Pre-trained Encoder:
     ```bash
     python task2-ImageNet.py
     ```
      
# Source Code Package in PyTorch
src/
  - Dataset 1
  - Dataset 2
  - Dataset 3
  - README.md
  - task1.py
  - task2-ImageNet.py
  - task2-PreTrainedModel.py

# The datasets are obtained using the sources below<br/>
<ol>
  <li>Dataset 1: Colorectal Cancer Classification [<a href="https://zenodo.org/records/1214456">Original Dataset</a> | <a href="https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp">Project Dataset</a>]</li>
  <li>Dataset 2: Prostate Cancer Classification [<a href="https://zenodo.org/records/4789576">Original Dataset</a> | <a href="https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp">Project Dataset</a>]</li>
  <li>Dataset 3: Animal Faces Classification [<a href="https://www.kaggle.com/datasets/andrewmvd/animal-faces">Original Dataset
  </a> | <a href="https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp">Project Dataset</a>]</li>
</ol>

