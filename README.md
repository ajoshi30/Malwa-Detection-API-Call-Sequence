🛡️ Malware Classification using API Call Sequences
📌 Overview

This project focuses on malware family classification using Windows API call execution traces.
Each malware sample is represented as a sequence of API calls, and the goal is to classify it into one of 8 malware families.

We implemented and compared multiple machine learning and deep learning models to understand how different architectures capture behavioral patterns in malware.

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

Split dataset into:

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

Test Accuracy: ~15%

3️⃣ 1D Convolutional Neural Network

Embedding Layer

1D Convolution

Max Pooling

Fully Connected Layer

Test Accuracy: ~15%

4️⃣ Random Forest (Baseline Classical ML)

Frequency-based feature extraction

RandomForestClassifier

Training Accuracy: ~99%
(Note: Evaluated on training data; proper test evaluation required for fair comparison.)

📊 Visualizations

The following plots were generated:

Histogram of API sequence length distribution

Heatmap of API correlation patterns

Bar chart of class distribution

Bar chart of most frequent API calls

Loss vs Epoch curve (to analyze learning behavior)

📈 Observations

Random baseline accuracy for 8 classes ≈ 12.5%

LSTM achieved ~32%, indicating meaningful pattern learning

GRU and CNN showed near-random performance

Malware families share overlapping API behaviors, making classification challenging

The current models represent baseline implementations without heavy hyperparameter tuning

🔬 Why Accuracy is Moderate

Malware API traces often share common system calls across families, resulting in:

High behavioral overlap

Weak discriminative sequential patterns

Noisy execution traces

Future improvements may include:

Hyperparameter tuning

Deeper architectures

Attention mechanisms

Feature engineering for classical models

Class imbalance handling

🚀 Future Work

Fine-tune deep learning models

Perform proper cross-validation

Evaluate RandomForest/XGBoost on test data

Add confusion matrix and per-class precision/recall

Experiment with Transformer-based sequence modeling

🛠️ Tech Stack

Python

PyTorch

Scikit-learn

NumPy

Matplotlib / Seaborn

📌 Conclusion

This project demonstrates that sequential deep learning models can learn behavioral patterns from malware API traces, though performance depends heavily on tuning and feature engineering.

The LSTM model successfully performed above random baseline, validating the viability of behavioral sequence-based malware classification.
