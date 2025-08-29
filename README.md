# Crop Prediction App

## Overview

The Crop Prediction App is designed to assist farmers and agricultural enthusiasts in predicting the best crops to plant based on environmental factors such as phosphorus levels, nitrogen levels, humidity, temperature, and rainfall.

## Features

- Input parameters for phosphorus, nitrogen, humidity, temperature, and rainfall.
- Predicts suitable crops based on the provided conditions.
- User-friendly interface to easily input data and view predictions.
- Built with robust algorithms to ensure accurate predictions.
- **Model accuracy:** The Gaussian Naive Bayes classifier used in this app achieves an accuracy of **[98]%** on the test dataset.

## Why Gaussian Naive Bayes Classifier?

We use the Gaussian Naive Bayes classification algorithm for crop prediction because:

- **Simplicity and Speed:** Gaussian Naive Bayes is easy to implement and computationally efficient, making it suitable for real-time predictions.
- **Effectiveness with Small Datasets:** It performs well even with relatively small datasets and is less prone to overfitting.
- **Handles Continuous Data Well:** Gaussian Naive Bayes assumes that features follow a normal (Gaussian) distribution, making it particularly effective for continuous input features such as those in this app.
- **Probabilistic Approach:** It provides the probability of each crop being suitable, allowing for more informed decision-making.

## Technologies Used

- Python 
- Machine Learning libraries (Scikit-learn)
- Flask 
- Pandas / NumPy (for data manipulation)

## Installation

Follow these steps to install and run the Crop Prediction App on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/hrishabh-dev/croppredictionflaskapp.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd croppredictionapp
   ```

3. **Install the required dependencies:**

   If using Python, you might run:

   ```bash
   pip install -r requirements.txt
   ```

   or for Node.js:

   ```bash
   npm install
   ```

4. **Run the application:**

   - For Python (Flask/Django):

     ```bash
     python app.py
     ```

   - For Node.js:

     ```bash
     npm start
     ```

5. **Access the app in your browser:**

   Open your browser and go to `http://localhost:5000` (or whichever port your app is hosted on).

## Usage

1. **Input Parameters:**

   Enter the following inputs in the provided fields:
   - **Phosphorus:** [Value in mg/kg]
   - **Nitrogen:** [Value in mg/kg]
   - **Humidity:** [Value in percentage]
   - **Temperature:** [Value in Celsius]
   - **Rainfall:** [Value in mm]

2. **Predict Crop:**

   After entering the values, click on the "Predict" button to receive crop recommendations.

3. **View Results:**

   The app will display a list of potential crops suitable for the given environmental conditions.

## Contributing

Contributions are welcome! If you would like to contribute to the Crop Prediction App, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to the branch and create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Machine Learning Libraries](https://scikit-learn.org/) for providing powerful tools for predictions.
- [Flask](https://flask.palletsprojects.com/) for creating the web application.

## Contact

For questions or feedback, please reach out to [hrishabh068@gmail.com].
