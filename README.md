# ACADEMICS
## Parametrically Optimized Automated Diagnostic System for Heart Disease Prediction using Deep Neural Networks

### Project Overview
This project aims to predict heart disease using deep learning models, leveraging patient data for binary classification. Two types of models, a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN), are utilized to classify heart disease based on medical parameters such as age, sex, chest pain type, resting blood pressure, and more.

### Key Features
- **CNN Model**: Uses 1D convolutional layers to extract features from structured data.
- **RNN Model**: Utilizes sequential data processing to detect patterns over time.
- **Data Preprocessing**: Includes reshaping and encoding target variables for compatibility with neural networks.
- **Evaluation Metrics**: Metrics such as accuracy, precision, recall, F1-score, and confusion matrices are used to evaluate model performance.
- **Model Comparison**: Visual comparisons of CNN and RNN performances are presented.

### Dataset
The dataset includes health-related parameters:
- Age
- Gender
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Maximum heart rate achieved (thalach)
- Exercise-induced angina (exang)
- Additional parameters

Data is split into training and testing sets for model evaluation.

### Models

1. **Convolutional Neural Network (CNN)**
   - Two convolutional layers with max pooling and dropout for regularization.
   - Activation: ReLU in hidden layers and Sigmoid in the output layer.
   - Validation Accuracy: ~93.5%

2. **Recurrent Neural Network (RNN)**
   - Simple RNN layers with dropout for regularization.
   - Activation: ReLU in hidden layers and Sigmoid in the output layer.
   - Validation Accuracy: ~98.5%

### Model Training
- **Batch Size**: 16
- **Epochs**: 500
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

### Visualization
Training accuracy and loss are plotted to analyze model convergence. Confusion matrices for both models are provided to display performance on the test data.

### Results
- The **CNN Model** achieved a validation accuracy of 93.5%.
- The **RNN Model** achieved a validation accuracy of 98.5%.
- The **RNN model** outperformed the CNN in precision and recall, making it more effective for heart disease prediction.

### Requirements
- Python 3.x
- TensorFlow 2.x
- Numpy
- Pandas
- Matplotlib
- Seaborn

### Usage
1. Clone the repository.
2. Run the Jupyter notebooks for preprocessing and model training:
   - `1_Preprocessing_File.ipynb`
   - `2_ModelTrainingFile.ipynb`
3. Evaluate the models using the provided test data.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

---
