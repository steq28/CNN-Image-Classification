# CNN Image Classification

PyTorch implementation of a convolutional neural network (CNN) for image classification using the CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 different classes:

- Airplanes
- Automobiles
- Birds
- Cats
- Deer
- Dogs
- Frogs
- Horses
- Ships
- Trucks

### Data Preprocessing
- Images are normalized to have a mean of 0 and a standard deviation of 1.
- The training data is split into training and validation sets.

## Model Architecture

The model is a convolutional neural network built with the following components:
- Convolutional layers
- Pooling layers
- Activation functions
- Fully connected layers

The architecture is optimized to achieve high accuracy on image classification tasks.

## Training and Evaluation

The model is trained using a pipeline that includes:
- Logging of training and validation loss/accuracy.
- Hyperparameter tuning to improve performance.

**Key Features**:
- Learning rate: ~0.03
- Optimizer: Stochastic Gradient Descent (SGD)
- Batch size: 32
- Epochs: 4

To prevent overfitting and improve accuracy, techniques such as Dropout, Batch Normalization, and data augmentation are utilized.

## Results

- **Best Test Accuracy Achieved**: > 80%

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/steq28/CNN-Image-Classification.git
   cd CNN-Image-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python main.py
   ```
