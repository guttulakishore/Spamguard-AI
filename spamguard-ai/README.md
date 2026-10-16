# 🛡️ SpamGuard AI - Smart Spam Message Detector

> Detect suspicious messages in seconds.

SpamGuard AI is a machine-learning web app built with **Python, Streamlit and scikit-learn**.
Paste any SMS, WhatsApp message or email text and it classifies it as **SPAM** or **NOT SPAM (HAM)**,
shows a model confidence score, and highlights suspicious phrases (links, urgency, prize claims,
requests for sensitive information).

> ⚠️ The model can make mistakes. **NOT SPAM never means guaranteed safe.**

---

## Features

- Real ML model: **TF-IDF + Multinomial Naive Bayes** trained on the UCI SMS Spam Collection
- Message analyzer with result card (🚨 SPAM / ✅ NOT SPAM), confidence and spam score
- Smart Detection Insights: highlighted indicators + the words that influenced the model
- 4 clickable sample messages that are analysed instantly
- Clear button, input validation (empty / too long messages)
- Model Performance dashboard: accuracy, precision, recall, F1, confusion matrix,
  train/test sizes, spam vs ham distribution (all from the real test set)
- Light and dark theme friendly UI with sidebar navigation (Home, Model Performance, About)
- Privacy: nothing is sent to external services, URLs are never opened, messages are never stored

## Project structure

    spamguard-ai/
    ├── app.py              # Streamlit interface (loads the trained model)
    ├── train_model.py      # Downloads/loads data, trains, evaluates, saves the model
    ├── requirements.txt    # Python dependencies
    ├── README.md           # This file
    ├── data/
    │   └── README.md       # How to get the dataset
    └── models/
        └── spam_classifier.joblib   # Created when you run train_model.py

## Setup on Windows (step by step)

Open **Command Prompt** or **PowerShell** inside the `spamguard-ai` folder.
You need Python 3.9 or newer (check with `python --version`).

1. Create a virtual environment

       python -m venv venv

2. Activate it

   Command Prompt:

       venv\Scripts\activate

   PowerShell:

       venv\Scripts\Activate.ps1

3. Install dependencies

       pip install -r requirements.txt

4. Download and prepare the dataset, then 5. train the model (one command does both)

       python train_model.py --download

   (Prefer manual download? See `data/README.md`, then run `python train_model.py`.)

6. Launch the app

       streamlit run app.py

   Your browser opens at http://localhost:8501.

## How it works (architecture)

    Dataset (SMS Spam Collection)
            │  train_model.py
            ▼
    Clean data → remove missing values, bad labels, duplicates
            ▼
    Train/test split (80/20, stratified)
            ▼
    Pipeline: TF-IDF Vectorizer → Multinomial Naive Bayes   (fitted on TRAIN data only)
            ▼
    Evaluate on TEST data → accuracy, precision, recall, F1, confusion matrix
            ▼
    Save pipeline + metrics → models/spam_classifier.joblib
            │  app.py
            ▼
    Streamlit app loads the file → predicts for each new message

**No data leakage:** the TF-IDF vectorizer and the classifier are in one scikit-learn `Pipeline`
and are fitted only on the training split. The test split is only used to measure performance.

**Explaining a decision:** for each word in your message the app multiplies its TF-IDF weight by how
much more likely the word is in spam than in ham (taken from the trained Naive Bayes model). The
words with the biggest effect are shown. The coloured highlights come from simple pattern rules
and are only *indicators* - the ML model makes the actual decision.

## Explaining the project in simple words (for presenting)

- *Problem:* spam messages waste time and can be scams.
- *Idea:* instead of writing rules by hand, we show a computer thousands of labelled examples
  and let it learn which words usually appear in spam.
- *TF-IDF:* turns a message into numbers by giving more weight to meaningful, less common words.
- *Naive Bayes:* a fast probability-based classifier. It asks "how likely is a message with these
  words to be spam versus normal?" and picks the more likely answer.
- *Testing:* we keep 20% of the data hidden during training and test on it, so the scores are honest.
- *Honesty:* the model is not perfect, so the app never promises a message is safe.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Install Python from python.org and tick **Add Python to PATH**, then reopen the terminal. Try `py` instead of `python`. |
| PowerShell: *running scripts is disabled* when activating | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and `venv\Scripts\activate`. |
| `'streamlit' is not recognized` | The environment is not active, or install failed. Activate the venv and run `pip install -r requirements.txt`. Or run `python -m streamlit run app.py`. |
| `pip install` fails / SSL or network errors | Check your internet, try `python -m pip install --upgrade pip`, then install again. |
| `Could not download the dataset automatically` | Your network may block the link. Download the dataset manually (see `data/README.md`) and run `python train_model.py`. |
| `No dataset found` | The file must be in `data/` and named `SMSSpamCollection` or `spam.csv`, or pass `--data "path"`. |
| `Could not find 'label' and 'text' columns` | Your CSV has different column names. Rename them to `v1` and `v2` (or `label` and `text`). |
| App shows "Model file not found" | You have not trained yet. Run `python train_model.py --download`. |
| App says it could not load the model | scikit-learn version changed. Run `python train_model.py` again to recreate it. |
| Port already in use | Run `streamlit run app.py --server.port 8502`. |

## Limitations

- Trained on English SMS messages from a public dataset; may perform worse on emails, other languages,
  slang, or new scam styles.
- Naive Bayes looks at words, not meaning or the sender, so clever spam can slip through and unusual
  normal messages can be flagged.
- The confidence value is the model's own score, not a guaranteed chance of being correct.
- Highlighted indicators are simple pattern rules, not proof of fraud.

## Future improvements

- Try other models (Logistic Regression, SVM, transformer models) and compare results
- Add multilingual support and email-specific training data
- Show a feedback button ("was this correct?") with explicit opt-in storage
- Add URL reputation checks (opt-in, clearly documented)
- Deploy online (Streamlit Community Cloud) and add automated tests

## Privacy

Messages stay on your computer. The app does not call external APIs, does not open links found in
messages, and does not store submitted messages. (The only network use is the optional one-time
dataset download in `train_model.py --download`.)
