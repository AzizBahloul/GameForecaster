 
# GameForecaster

`GameForecaster` is a Python library designed to predict the success of video games based on their features using machine learning models. This package leverages pre-trained models, enabling developers to make predictions without needing to understand the intricacies of machine learning or data preprocessing.

## Features

- **Predict Game Success**: Classify whether a game will be successful based on its attributes.
- **Pre-trained Models**: Use the provided models, scalers, and label encoders without needing to train your own.
- **Easy Integration**: Simple API design for seamless integration into your existing projects.

## Installation

You can install `GameForecaster` from PyPI using pip:

```bash
pip install GameForecaster
```

Make sure to have the following files in your project directory:
- `model.pkl`: The trained model used for making predictions.
- `scaler.pkl`: The scaler used for normalizing numerical features.
- `label_encoders.pkl`: The label encoders used for categorical features.

These files should be placed in the directory where you will use the library, or you should provide the paths to these files when initializing the `GameForecaster` class.

## Usage

Here’s how to use the `GameForecaster` library to predict the success of a game:

### Step 1: Import the Library

First, import the necessary classes from the `GameForecaster` library.

```python
from game_forecaster import GameForecaster, GameFeatures
```

### Step 2: Initialize the Predictor

Create an instance of the `GameForecaster` class. You will need to provide the paths to your model artifacts.

```python
# Initialize the GameForecaster with paths to your model files
predictor = GameForecaster(model_path='path/to/model.pkl',
                            scaler_path='path/to/scaler.pkl',
                            label_encoders_path='path/to/label_encoders.pkl')
```

### Step 3: Prepare Game Features

Create an instance of the `GameFeatures` class with the required game attributes. Ensure to follow the specifications for each attribute, including types and value constraints.

```python
# Prepare game features for prediction
game = GameFeatures(
    Year_of_Release=2023,
    Genre="Action",
    Publisher="Publisher Name",
    NA_Sales=1.5,
    EU_Sales=0.7,
    JP_Sales=0.3,
    Other_Sales=0.2,
    Global_Sales=2.7,
    Critic_Score=85,
    Critic_Count=50,
    User_Score=8.5,
    User_Count=1500,
    Developer="Developer Name",
    Rating="E"
)
```

### Step 4: Make a Prediction

Use the `predict` method of the `GameForecaster` instance to predict the success of the game.

```python
# Make a prediction
prediction_result = predictor.predict(game)

# Display the prediction result
print(f"Prediction: {'Successful' if prediction_result.prediction else 'Not Successful'}")
print(f"Confidence: {prediction_result.confidence:.2f}")
print(f"Feature Importance: {prediction_result.feature_importance}")
print(f"Explanation: {prediction_result.explanation}")
```

### Example Output

Assuming the model predicts successfully, you might see output similar to:

```
Prediction: Successful
Confidence: 0.78
Feature Importance: {'Genre': 0.15, 'Publisher': 0.20, 'NA_Sales': 0.25, ...}
Explanation: The model predicts that the game will be successful with 78.0% confidence. This prediction is primarily based on Genre (15.0%), Publisher (20.0%), and NA_Sales (25.0%).
```

### Batch Predictions

You can also predict the success of multiple games at once using batch processing:

```python
# Create a list of game features for batch prediction
batch_games = [
    GameFeatures(
        Year_of_Release=2022,
        Genre="RPG",
        Publisher="Publisher A",
        NA_Sales=0.5,
        EU_Sales=0.4,
        JP_Sales=0.6,
        Other_Sales=0.1,
        Global_Sales=1.6,
        Critic_Score=75,
        Critic_Count=30,
        User_Score=7.5,
        User_Count=800,
        Developer="Developer A",
        Rating="T"
    ),
    GameFeatures(
        Year_of_Release=2021,
        Genre="Adventure",
        Publisher="Publisher B",
        NA_Sales=2.0,
        EU_Sales=1.0,
        JP_Sales=0.5,
        Other_Sales=0.4,
        Global_Sales=4.0,
        Critic_Score=90,
        Critic_Count=60,
        User_Score=9.0,
        User_Count=2000,
        Developer="Developer B",
        Rating="M"
    ),
    # Add more GameFeatures as needed
]

# Make batch predictions
batch_predictions = predictor.batch_predict(batch_games)

# Display the batch prediction results
for result in batch_predictions:
    print(f"Prediction for {result['game_name']}: {'Successful' if result['prediction'] else 'Not Successful'}")
```

## Conclusion

`GameForecaster` is an effective tool for game developers and analysts seeking to make data-driven predictions about the success of their video games. By utilizing pre-trained models, it eliminates the need for extensive data science knowledge, making it accessible and easy to implement in any Python application.

For further details, please refer to the [documentation](https://pypi.org/project/GameForecaster/) or visit the project's GitHub repository for source code and contributions.

 
