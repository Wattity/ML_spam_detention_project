# Spam Detection Project

A machine learning application that classifies incoming text as either spam or not spam. The project combines a FastAPI backend for prediction and a Streamlit frontend for user interaction, making it easy to test message classification in a simple web interface.

## Overview

This project demonstrates a text classification workflow for spam detection using natural language preprocessing and Word2Vec-based message embeddings. An input message is cleaned, tokenized, and converted into a vector representation, which is then passed to a trained classifier model to determine whether it is spam or legitimate.

The solution includes:

- A preprocessing pipeline for text normalization
- Word2Vec model to generate semantic embeddings
- A trained machine learning classifier
- A REST API for predictions
- A Streamlit dashboard for easy interaction

## Features

- Detects spam messages from free-form input text
- Supports real-time prediction via REST API
- Provides a simple user-friendly Streamlit UI
- Uses cleaned and normalized text before vectorization
- Includes model artifacts and preprocessing logic for re-use

## Project Structure

```text
spam_detention_project/
├── LICENSE
├── README.md
├── requirements.txt
├── backend/
│   ├── main.py                  # FastAPI service
│   ├── preprocessing.py         # Text preprocessing and vectorization
│   ├── spam_model.joblib        # Trained model
│   └── spam_word2vec.model     # Word2Vec model
├── frontend/
│   └── app.py                   # Streamlit app
├── notebook/
│   └── Untitled1.ipynb         # Experimental notebook
└── venv/                        # Local Python virtual environment
```

## Tech Stack

- Python 3.x
- FastAPI
- Streamlit
- NumPy
- pandas
- scikit-learn
- Gensim
- NLTK
- joblib

## How the Model Works

1. The user enters a message in the app or sends a request to the API.
2. The backend preprocesses the text:
   - lowercasing
   - tokenization
   - stopword removal
   - punctuation cleanup
   - stemming and lemmatization
   - whitespace normalization
3. The cleaned text is transformed into a vector representation using a Word2Vec model.
4. The resulting vector is fed into the trained classifier.
5. The system returns:
   - "Spam"
   - "Not spam"

## Requirements

Python dependencies are listed in [requirements.txt](requirements.txt).

Install them with:

```bash
pip install -r requirements.txt
```

### NLTK Data

The project uses NLTK resources such as stopwords, punkt, and wordnet. These are downloaded automatically in the preprocessing module, but you can also download them manually if needed:

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd spam_detention_project
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Backend API

Navigate to the backend folder and start the FastAPI server:

```bash
cd backend
python main.py
```

The API will run on:

```text
http://127.0.0.1:8000
```

### Available endpoints

#### Home

```http
GET /home
```

Response:

```json
{
  "message": "Welcome to the spam detection API"
}
```

#### Prediction

```http
POST /predict
```

Request body:

```json
{
  "message": "Congratulations! You have won a free prize. Claim now!"
}
```

Example response:

```json
{
  "prediction": "Spam"
}
```

## Running the Streamlit Frontend

Open a new terminal and start the frontend app:

```bash
cd frontend
streamlit run app.py
```

This opens the app in the browser, where you can type a message and click the prediction button to classify it.

## Example Usage

### API Example with curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hi John, can we meet tomorrow at 3 PM?"}'
```

Example output:

```json
{
  "prediction": "Not spam"
}
```

### UI Flow

1. Enter a message like a promotional SMS, scam text, or normal conversation.
2. Click the Predict Message button.
3. The app displays the predicted label.

## Project Files

### backend/preprocessing.py

Handles text cleaning and vector generation.

Key functions:

- `preprocess_text_pipeline(text)`
  - normalizes and cleans input strings
- `get_message_vector(text, word2vec_model)`
  - converts a processed message into a Word2Vec average embedding

### backend/main.py

Defines the FastAPI application and prediction endpoint.

Key behavior:

- loads the trained classifier from `spam_model.joblib`
- loads the `Word2Vec` model from `spam_word2vec.model`
- preprocesses the incoming message
- produces a single vector embedding
- returns a classified result

### frontend/app.py

Provides a simple Streamlit interface for users to interact with the backend API.

## Notes and Limitations

- This project is intended as a lightweight demo and educational example.
- Prediction quality depends on the quality and volume of the training data.
- The app currently uses a local Word2Vec model and joblib classifier, so the backend must be available when running the frontend.
- If you move project files or run the app from a different working directory, ensure the model paths still resolve correctly.

## Troubleshooting

### API returns an error or model not found

Check that you are running the backend from the correct folder and that the following files exist in `backend/`:

- `spam_model.joblib`
- `spam_word2vec.model`

### NLTK resource errors

Run:

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
```

### Streamlit cannot connect to the backend

Ensure the API is running before launching the frontend.

Check that the frontend is making requests to:

```text
https://ml-spam-detention-project.onrender.com/docs
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contribution

Contributions are welcome. If you want to improve the model, add more preprocessing steps, or build a better frontend experience, feel free to open an issue or submit a pull request.
