# Happiness Index Predictor

This project was developed during a 24-hour hackathon and aims to predict the happiness index of users through two approaches: manual questionnaire responses and facial image recognition. The application provides personalized recommendations to uplift the user's happiness index and mood based on the predictions.

## Features

### 1. Manual Questionnaire Approach:
- Users can fill out a questionnaire covering various aspects of life such as work, relationships, health, etc.
- Machine learning models trained on global happiness index data analyze the responses to predict the user's happiness index.
- Hardcoded recommendations are provided based on the predicted happiness index to improve the user's happiness in different life aspects.

### 2. Facial Image Recognition Approach:
- Users answer personal questions such as hobbies and coping mechanisms for sadness.
- Users upload a facial image, and deep learning models recognize the user's mood.
- Recommendations are given based on the predicted mood to uplift the user's mood, such as suggesting activities or relaxation techniques.

## Instructions to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/soham1826/Happiness-Index-Predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd happiness-index-predictor
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open your web browser and go to [http://localhost:8501](http://localhost:8501) to access the application.

## Technologies Used

- Python
- Streamlit
- Machine Learning (scikit-learn)
- Deep Learning (TensorFlow, Keras)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the [MIT License](LICENSE).

