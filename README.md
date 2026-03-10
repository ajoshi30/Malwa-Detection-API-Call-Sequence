🛡️ Malware Classification using API Call Sequences
📌 Overview

This project focuses on malware family classification using Windows API call execution traces. Each malware sample is represented as a sequence of API calls, and the goal is to classify it into one of 8 malware families.

We implemented and compared multiple machine learning and deep learning models and later applied hyperparameter tuning to improve deep learning performance and analyze how different architectures capture behavioral patterns in malware.

📂 Dataset Description

Total Samples: 7107
Number of Classes: 8

Classes:

Adware

Backdoor

Downloader

Dropper

Spyware

Trojan

Virus

Worms

Each sample consists of:

A sequence of numeric API call IDs (representing execution behavior)

A corresponding malware family label

⚙️ Preprocessing Pipeline

Loaded API call sequences from text file

Encoded malware labels using LabelEncoder

Padded sequences to fixed length (500)

Split dataset into

80% Training

20% Testing (stratified)

Calculated vocabulary size for embedding layer

🧠 Models Implemented
1️⃣ LSTM (Long Short-Term Memory)

Embedding Layer

LSTM Layer

Fully Connected Layer

CrossEntropy Loss

Test Accuracy: ~32%

2️⃣ GRU (Gated Recurrent Unit)

Embedding Layer

GRU Layer

Fully Connected Layer

Test Accuracy: ~31%

3️⃣ 1D Convolutional Neural Network

Embedding Layer

1D Convolution

Max Pooling

Fully Connected Layer

Test Accuracy: ~33%

4️⃣ Random Forest (Baseline Classical ML)

Frequency-based feature extraction

RandomForestClassifier

Training Accuracy: ~99%
(Note: Evaluated on training data, used as baseline)

🔧 Hyperparameter Tuning (LSTM)

We performed grid search hyperparameter tuning on the LSTM model using different values of:

Embedding dimension

Hidden dimension

Learning rate

Batch size

Tested combinations:

embed_dim = [64, 128]
hidden_dim = [64, 128]
lr = [0.001, 0.0005]
batch_size = [32, 64]

Best parameters found:

embed_dim = 128
hidden_dim = 64
lr = 0.001
batch_size = 32

Best accuracy during tuning: ~38%

Final model trained with more epochs (20):

Final Tuned Accuracy: ~41%

📊 Visualizations

Generated plots:

Histogram of API sequence length distribution

Heatmap of API correlation patterns

Bar chart of class distribution

Bar chart of most frequent API calls

Confusion matrix

Hyperparameter tuning accuracy plot

Training accuracy curve

📈 Observations

Random baseline for 8 classes ≈ 12.5%

Initial LSTM accuracy ≈ 32%

After hyperparameter tuning ≈ 38%

Final tuned model ≈ 41%

GRU and CNN performed similarly to LSTM but slightly lower

RandomForest showed very high training accuracy but not reliable for test comparison

Malware families share overlapping API behavior, making classification difficult

Hyperparameter tuning significantly improved performance.

🔬 Why Accuracy is Moderate

Malware API traces often contain similar system calls across families:

High behavioral overlap

Long noisy sequences

Weak class-specific patterns

Sequence models need tuning to capture meaningful behavior.

🚀 Future Work

More hyperparameter tuning

BiLSTM / Attention / Transformer models

Proper cross-validation

Test evaluation for classical ML models

Class imbalance handling

Feature engineering for API patterns

🛠️ Tech Stack

Python
PyTorch
Scikit-learn
NumPy
Matplotlib
Seaborn

📌 Conclusion

This project demonstrates that sequential deep learning models can learn behavioral patterns from malware API traces.

Baseline LSTM achieved ~32% accuracy.
After hyperparameter tuning, performance improved to ~41%, showing that model performance depends strongly on proper tuning and training configuration.

The results confirm that behavioral sequence-based malware classification is possible, but requires careful optimization and feature design.
