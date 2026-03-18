# Loan Approval Predictor

Machine Learning based **Loan Approval Prediction** system. This project
generates a realistic synthetic dataset (\~400 rows), trains multiple ML
models, selects the best model, and exposes a simple web UI (Flask)
for making predictions.

------------------------------------------------------------------------

## Quick start

Clone the repository from GitHub, then run the commands **in the order
shown**.

``` bash
git clone https://github.com/amankr-07/loan-approval-predictor
cd loan-approval-predictor
```

1.  Install dependencies

``` bash
pip install -r requirements.txt
```

2.  Generate the training dataset

``` bash
python -m data.generate_dataset
```

3.  Train the machine learning models

``` bash
python -m models.train_model
```

4.  Start the application

``` bash
python app.py
```

After step 4, a `localhost` link will appear in the terminal. Open that
link in any browser.

> ⚠️ Make sure your device is connected to the **internet the first
> time** you open the app in any browser.

------------------------------------------------------------------------

## Project structure

    loan-approval-predictor/
    ├── app.py                      # Flask app
    ├── requirements.txt
    ├── README.md
    ├── data/
    │   ├── generate_dataset.py     # creates ~400 rows realistic dataset
    │   └── loan_data.csv
    ├── models/
    │   ├── train_model.py          # trains multiple models and saves best model
    │   └── model.pkl, scaler.pkl
    ├── utils/
    │   └── preprocess.py
    ├── static/
    │   └── style.css
    ├── utils/
    │   └── index.html
    │   └── result.html
    └── notebooks/
        └── analysis.ipynb

------------------------------------------------------------------------

## Features

-   Synthetic dataset generation (\~400 rows)
-   Multiple machine learning models trained
-   Automatic best model selection
-   Simple UI for user prediction
-   Real-time loan approval prediction

------------------------------------------------------------------------

## Notes

-   You can increase the dataset size by modifying the row count inside
    `data/generate_dataset.py`.
-   Re-run dataset generation and model training if you change dataset
    size.
    