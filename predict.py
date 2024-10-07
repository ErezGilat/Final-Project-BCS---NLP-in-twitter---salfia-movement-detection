import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Function to load the two models
def load_models(model1_path, model2_path):
    """ Load the two models from .h5 files """
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    return model1, model2

# Function to preprocess the tweets
def preprocess_tweets(tokenizer, tweets, max_length):
    """ Tokenize and pad the input tweets """
    sequences = tokenizer.texts_to_sequences(tweets)  # Convert tweets to sequences of integers
    padded_sequences = pad_sequences(sequences, maxlen=max_length)  # Pad sequences to max_length
    return padded_sequences

# Function to make predictions with both models and average the results
def predict(models, tweets):
    """ Predict labels for the input tweets using both models and average the predictions """
    model1, model2 = models
    
    predictions_model1 = model1.predict(tweets)  # Predictions from model 1
    predictions_model2 = model2.predict(tweets)  # Predictions from model 2
    
    # Averaging the predictions
    average_predictions = (predictions_model1 + predictions_model2) / 2
    return np.round(average_predictions).astype(int)  # Assuming binary classification, round predictions

# Main function to handle the prediction process
def main(args):
    # Read the input CSV file containing the tweets
    input_data = pd.read_csv(args.input_csv)
    
    # Assuming the CSV has a column 'tweet' with the tweet text
    tweets = input_data['tweet'].values
    
    # Load the tokenizer from a saved file (you need to provide the tokenizer if required)
    tokenizer = Tokenizer()  # Initialize your tokenizer
    tokenizer.fit_on_texts(tweets)  # You may need to load your own pre-trained tokenizer instead of fitting on this data
    
    max_length = 100  # Define the max length of sequences (adjust according to your training)
    tokenized_tweets = preprocess_tweets(tokenizer, tweets, max_length)
    
    # Load the two models
    model1, model2 = load_models(args.model1, args.model2)
    
    # Make predictions by averaging the output from both models
    predictions = predict((model1, model2), tokenized_tweets)
    
    # Save the results into a new CSV file
    output_df = pd.DataFrame({
        'tweet': input_data['tweet'],
        'predicted_label': predictions.flatten()  # Ensure the predictions are saved in a single column
    })
    
    output_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

# Entry point for the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels for tweets using two models and average the predictions.")
    
    # Define the required arguments for input and output
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file with tweets')
    parser.add_argument('--model1', type=str, required=True, help='Path to the first model (.h5 file)')
    parser.add_argument('--model2', type=str, required=True, help='Path to the second model (.h5 file)')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file where predictions will be saved')
    
    args = parser.parse_args()
    main(args)
