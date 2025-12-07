# Precision Fracture Classification: Leveraging Transfer Learning for Enhanced Diagnostics

This repository contains code for a deep learning model that classifies bone fractures from X-ray images using transfer learning techniques. The project demonstrates the application of pre-trained convolutional neural networks (CNNs) to achieve high accuracy in medical image classification.

## Overview

Bone fracture detection is crucial in medical diagnostics. This project leverages transfer learning with state-of-the-art pre-trained models to classify X-ray images into "Fractured" and "Not Fractured" categories. The implementation includes data preprocessing, model training, evaluation, and visualization.

## Features

- **Transfer Learning**: Utilizes pre-trained models including EfficientNetB2, VGG16, and ResNet50
- **Data Augmentation**: Implements image preprocessing and augmentation for robust training
- **Model Evaluation**: Comprehensive evaluation with confusion matrices, classification reports, and accuracy metrics
- **Visualization**: Plots training history, validation metrics, and sample predictions
- **GPU Support**: Automatic detection and utilization of GPU for faster training

## Models Used

1. **EfficientNetB2**: A highly efficient CNN architecture that scales well with available resources
2. **VGG16**: A deep convolutional network known for its simplicity and effectiveness
3. **ResNet50**: A residual network that addresses the vanishing gradient problem in deep networks

## Dataset

The project expects a dataset organized in the following structure:
```
BoneFracture/
├── train/
│   ├── fractured/
│   └── not_fractured/
├── test/
│   ├── fractured/
│   └── not_fractured/
└── val/
    ├── fractured/
    └── not_fractured/
```

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV
- Scikit-learn

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Racheal-Akinbo/Precision-Fracture-Classification-Leveraging-Transfer-Learning-for-Enhanced-Diagnostics.git
   cd Precision-Fracture-Classification-Leveraging-Transfer-Learning-for-Enhanced-Diagnostics
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. For Google Colab usage, mount your Google Drive and update the dataset paths in the notebook.

## Usage

1. Open the `bone_fracture_classification.ipynb` notebook in Jupyter or Google Colab.

2. Update the dataset paths to point to your local directories or Google Drive folders.

3. Run the cells sequentially to:
   - Import libraries
   - Load and preprocess data
   - Visualize the dataset
   - Train models (EfficientNet, VGG16, ResNet50)
   - Evaluate model performance

### Key Functions

- `data(dataset_path)`: Loads image paths and labels from directory structure
- `configure_model(model)`: Configures pre-trained models for binary classification
- `model(new_model, layers_num, trainable)`: Freezes/unfreezes model layers
- `scheduler(epoch, lr)`: Learning rate scheduler
- `callbacks(my_model, patience)`: Sets up training callbacks

## Training Configuration

- **Image Size**: 260x260 pixels
- **Batch Size**: 32
- **Epochs**: 7 (with early stopping)
- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Results

The models achieve high accuracy on the test set. Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Sample results from the notebook show competitive performance across all three models.

## Model Saving and Loading

Trained models are saved in HDF5 (.h5) and Keras (.keras) formats. The notebook includes code to load and evaluate saved models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Dataset source: kaggle
- Pre-trained models from TensorFlow/Keras applications
- EfficientNet implementation by the authors

## Contact

For questions or suggestions, please open an issue on GitHub.
