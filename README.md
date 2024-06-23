# Treatment Prediction API

## Overview

This API predicts the treatment type for breast cancer patients based on input features. The prediction is made using a pre-trained Random Forest model.

## Requirements

- Python 3.6+
- Flask
- Flask-CORS
- joblib
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/treatment-prediction-api.git
   cd treatment-prediction-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the model file `model.pkl` in the project directory**

4. **Run the API**
   ```bash
   python app.py
   ```

## Endpoints

### Home

**URL:** `/`  
**Method:** `GET`  
**Description:** Returns a welcome message.

**Response:**
```json
"Welcome to the Treatment Prediction API!"
```

### Predict

**URL:** `/predict`  
**Method:** `POST`  
**Description:** Accepts a JSON object with features and returns a treatment prediction.

**Request:**

- **Headers:** `Content-Type: application/json`
- **Body:** A JSON object containing the features for prediction.

  ```json
  {
      "features": [1, 1, 0, 50]
  }
  ```

**Response:**

- **Success (200):**
  ```json
  {
      "prediction": "Treatment A"
  }
  ```
  where `prediction` is the treatment type predicted by the model.

- **Error (400):**
  ```json
  {
      "error": "Error message"
  }
  ```

## Usage

### Using Postman

1. **Open Postman**.

2. **Create a new POST request**.

3. **Set the URL** to `https://treatment-prediction-api.onrender.com/predict`.

4. **Go to the `Body` tab**, select `raw`, choose `JSON`, and add the JSON object:

    ```json
    {
        "features": [1, 1, 0, 50]
    }
    ```

5. **Send the request** and view the prediction in the response.

### Sample Code

Here's how you can use the API programmatically with Python's `requests` library:

```python
import requests

url = 'https://treatment-prediction-api.onrender.com/predict'
data = {
    "features": [1, 1, 0, 50]
}

response = requests.post(url, json=data)
print(response.json())
```

## Model Information

### Dataset Description

The dataset used for training the model contains information on breast cancer patients and their respective treatments. The key features used for prediction are:

- **ER (Estrogen Receptor Status)**
- **PR (Progesterone Receptor Status)**
- **HER2 (Human Epidermal growth factor Receptor 2)**
- **Ki67 (%)**

The target variable is the treatment type.

### Model Training

The model is a Random Forest Classifier trained with the following steps:

1. **Data Loading and Preprocessing:**
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = pd.read_excel("data/test.xlsx")
    X = data[['ER', 'PR', 'HER2', 'Ki67 (%)']]
    y = data['Treatment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Model Training:**
    ```python
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    ```

3. **Model Evaluation:**
    ```python
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    ```

4. **Feature Importance:**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    importances = rf.feature_importances_
    features = X.columns
    indices = importances.argsort()

    plt.figure(figsize=(8, 6))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    ```

5. **Confusion Matrix:**
    ```python
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    ```

6. **Treatment Distribution:**
    ```python
    treatment_counts = data['Treatment'].value_counts()

    plt.figure(figsize=(24, 6))
    sns.barplot(x=treatment_counts.index, y=treatment_counts.values)
    plt.title('Number of Patients by Treatment Type')
    plt.xlabel('Treatment Type')
    plt.ylabel('Number of Patients')
    plt.show()
    ```

7. **Save the Model:**
    ```python
    import joblib
    joblib.dump(rf, 'model.pkl')
    ```

### Model Usage

You can use the saved model to make predictions as follows:

```python
import joblib
import numpy as np

# Load the model
model = joblib.load('model.pkl')

# Make a prediction
input_data = np.array([[1, 1, 0, 50]])
prediction = model.predict(input_data)
print("Predicted treatment:", prediction)
```

## Conclusion

This API is a tool to predict the treatment type for breast cancer patients based on their clinical features. Ensure proper testing and validation before using it in a production environment.
