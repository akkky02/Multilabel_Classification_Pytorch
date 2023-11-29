# Multi-Label Text Classification with PyTorch

This notebook implements a neural network for multi-label text classification on the Stack Overflow tagged text dataset.

## Notebook Contents

The notebook covers the following:

**Data Preparation**
- Load the cleaned Stack Overflow text data
- Split the data into train/validation/test sets
- Encode the multiple labels using MultiLabelBinarizer

**Model Architecture**
- Embeddings → Dense Layer → ReLU → Dropout → BatchNorm → Dense Layer → ReLU → Dropout → BatchNorm → Output
- Use EmbeddingBag layer for computational efficiency
- Output layer with sigmoid activations to predict multiple labels 

**Training**
- Binary Cross Entropy loss function
- Adam optimizer
- Hamming loss to evaluate performance

**Results**
- Achieved a test Hamming Loss of 0.043 after 5 epochs of training
- Overfits after 3-4 epochs, dropout helps regularization 

**Conclusion**
- The model architecture works decently for multi-label classification
- There is some overfitting on this dataset
- More training data and fine-tuning hyperparameters could help
