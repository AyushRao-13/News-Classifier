# 📰 NewsSense AI — News Article Classification Using NLP and CNN

NewsSense AI is a deep learning application that classifies news headlines and articles into four categories: **World, Sports, Business, and Sci/Tech**. It uses Natural Language Processing (NLP), a Convolutional Neural Network (CNN), and a Streamlit interface to provide predictions and category probability estimates.

## 🚀 Features

- **News classification:** Predicts one of four news categories.
- **CNN v3 model:** Uses an embedding layer and parallel convolution filters to learn patterns in text.
- **Machine learning baseline:** Compares the CNN with TF-IDF + Logistic Regression.
- **Exploratory Data Analysis:** Examines category distribution, article lengths, and frequent words.
- **Evaluation dashboard:** Displays accuracy comparisons, a confusion matrix, and category-wise F1 scores.
- **Interactive interface:** Accepts custom text and provides sample headlines and probability visualizations.

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Natural Language Processing (NLP)
- Convolutional Neural Networks (CNN)
- Scikit-learn
- Pandas and NumPy
- Matplotlib
- Streamlit

## 📊 Dataset

The project uses the AG News dataset, containing four news categories:

| Category | Description |
|---|---|
| World | International news and global affairs |
| Sports | Sports events and competitions |
| Business | Markets, companies, and the economy |
| Sci/Tech | Science, technology, and innovation |

The final experiment uses **120,000 training articles** and **7,600 test articles**, as recorded in the project's evaluation results.

## 🏗️ Model Architecture

The CNN pipeline consists of:

1. **Text tokenization:** Converts text into sequences of vocabulary IDs using a fitted tokenizer.
2. **Padding and truncation:** Converts sequences to a consistent length for model input.
3. **Embedding layer:** Learns dense vector representations of tokens.
4. **Parallel convolution filters:** Uses kernel sizes 3, 5, and 7 to learn local text patterns at different scales.
5. **Classification layers:** Produce scores for the four news categories using a softmax output.

## 📈 Results

The final experiment produced the following results:

| Metric | CNN v3 | TF-IDF + Logistic Regression |
|---|---:|---:|
| Accuracy | 91.78% | 92.00% |
| Macro Precision | 91.76% | 91.99% |
| Macro Recall | 91.78% | 92.00% |
| Macro F1-score | 91.75% | 91.98% |

Both models were evaluated on the same test set. In this experiment, TF-IDF + Logistic Regression performed slightly better overall, while the CNN demonstrates how deep learning can be applied to text classification.

*These figures reflect the recorded experiment and should be regenerated if the models or evaluation setup change.*

## 🖥️ Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd news-classifier
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install tensorflow datasets scikit-learn pandas numpy matplotlib streamlit
```

### 4. Check model files

Ensure the following files are present:

```text
models/
├── news_cnn_v3.keras
└── tokenizer_v3.pkl
```

### 5. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed in your terminal.

## 📁 Project Structure

```text
news-classifier/
├── app.py
├── eda.py
├── evaluate.py
├── compare_models.py
├── compare_full_data.py
├── train.py
├── train_cnn_v2.py
├── train_cnn_v3.py
├── analyze_sequence_length.py
├── models/
│   ├── news_cnn_v3.keras
│   └── tokenizer_v3.pkl
└── results/
    ├── eda/
    ├── confusion_matrix.png
    ├── full_data_model_comparison.png
    ├── full_data_category_f1.png
    └── full_data_category_comparison.csv
```

*The structure above is illustrative; keep only files and directories that actually exist in your repository.*

## ⚠️ Limitations

- The model is designed for the four categories in AG News.
- Predictions can be less reliable for ambiguous, very short, or out-of-domain text.
- Model confidence represents an estimated probability, not a guarantee of correctness.
- Results depend on the training data, preprocessing, and evaluation setup.

## 🔮 Future Improvements

- Compare CNN performance with LSTM and transformer-based models.
- Evaluate generalization on additional news datasets.
- Improve handling of ambiguous and out-of-domain text.
- Deploy the application for public access.

## 👨‍💻 Project Information

**Project:** NewsSense AI — News Article Classification Using NLP and CNN  
**Application:** Streamlit  
**Task:** Multi-class text classification  
**Dataset:** AG News

Developed as an academic mini-project exploring NLP and deep learning.
