# Dataset folder

SpamGuard AI is trained on the **SMS Spam Collection** dataset (UCI Machine Learning Repository):
about 5,500 English SMS messages labelled `ham` (normal) or `spam`.

## Option 1 - automatic (recommended)

From the project folder:

    python train_model.py --download

This downloads the official zip from UCI, extracts `SMSSpamCollection` into this folder, and trains the model.

## Option 2 - manual

1. Open https://archive.ics.uci.edu/dataset/228/sms+spam+collection and click **Download**
   (the Kaggle version "SMS Spam Collection Dataset", `spam.csv`, also works).
2. Unzip and copy **one** of these files into this `data/` folder:
   - `SMSSpamCollection` (UCI tab-separated file), or
   - `spam.csv` (Kaggle CSV with columns `v1`, `v2`)
3. Run `python train_model.py`

Or point to a file anywhere on your computer:

    python train_model.py --data "C:\path\to\spam.csv"

## File formats understood

- UCI: each line is `label<TAB>message` (no header).
- CSV: columns `v1`/`v2` or `label`/`text` or `Category`/`Message`.
  Labels must be `ham` or `spam`; other rows are removed.

Dataset files are not included in this project - please download them from the official source.
