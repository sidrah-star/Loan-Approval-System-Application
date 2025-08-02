# Premium Loan Eligibility Predictor

An AI-powered web application that predicts loan eligibility based on user input, using a machine learning model trained on real loan data. The app provides instant results, confidence scores, and personalized tips to improve approval chances.

## Features
- Modern, responsive web interface for loan eligibility prediction
- Input validation and user-friendly form
- Instant prediction with confidence score
- Personalized tips to improve eligibility
- Sample profiles for quick demo
- REST API endpoint for programmatic access
- Model training notebook included

## Demo
![Screenshot 2025-06-20 193437](https://github.com/user-attachments/assets/01044d99-bdbd-49d8-9dd6-f399c2ddd9b4)
![Screenshot 2025-06-20 193510](https://github.com/user-attachments/assets/73516e87-8633-4495-84de-271791b1d54f)
![Screenshot 2025-06-20 193547](https://github.com/user-attachments/assets/bacbfedb-2fba-4c38-9a64-0718664698a2)
![Screenshot 2025-06-20 193607](https://github.com/user-attachments/assets/ffbb0460-5ec3-4a71-bcf5-03074d1a7a7b)
![Screenshot 2025-06-20 193637](https://github.com/user-attachments/assets/83a43ef3-c60b-472c-87cc-654645a325ed)
![Screenshot 2025-06-20 193659](https://github.com/user-attachments/assets/b9b6aa6e-c3cd-439f-9a81-8e9f7ab36eeb)



## Getting Started

### Prerequisites
- Python 3.8+

### Installation
1. Clone this repository:
   ```bash

   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the model file `loan_eligibility_model_tuned.pkl` is present in the root directory. If not, retrain using the notebook in `Model Traning Content/`.

### Running the App
```bash
python app.py
```
The app will be available at [http://localhost:5000](http://localhost:5000).

## Usage
### Web Interface
- Fill in your details in the form.
- Click **Predict Loan Eligibility** to get instant results.
- Use the sample buttons to auto-fill eligible or not-eligible profiles.
- View personalized tips and confidence score.

### API Usage
Send a POST request to `/api/predict` with JSON payload (see `app.py` for details). Example:
```json
{
  "Gender": 1,
  "Married": 1,
  "Dependents": 0,
  "Education": 0,
  "Self_Employed": 0,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": 2
}
```

## Model Training
- See `Model Traning Content/Model_Training.ipynb` for full training pipeline.
- Data: `train.csv` (public loan dataset)
- Preprocessing: missing value imputation, label encoding, feature engineering (total income, log loan amount)
- Model: XGBoost Classifier, hyperparameter tuning
- Accuracy: ~76% on test set
- Model saved as `loan_eligibility_model_tuned.pkl`

## Technologies Used
- Python, Flask, scikit-learn, XGBoost, NumPy, Pandas, Joblib
- HTML5, CSS3, JavaScript (for frontend)

## File Structure
```
loan_app/
  app.py                  # Flask app
  loan_eligibility_model_tuned.pkl  # Trained model
  requirements.txt        # Python dependencies
  templates/
    index.html            # Web UI
  Model Traning Content/
    Model_Training.ipynb  # Model training notebook
    train.csv             # Training data
```

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Credits
Developed by Shrikant Wadkar. Inspired by open-source ML and data science communities. 
