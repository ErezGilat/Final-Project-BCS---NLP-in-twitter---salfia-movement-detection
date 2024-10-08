# NLP Model for Identifying Salafi Jihadist Tweets in England

This repository contains the code and pre-trained models for the final project titled **"Developing a Natural Language Processing Model to Identify Tweets Associated with the Salafi Movement in England."** The project was completed as part of the requirements for a B.Sc. degree and is aimed at improving the detection and monitoring of extremist content on social media platforms.

## Project Overview

This project leverages state-of-the-art Natural Language Processing (NLP) techniques to develop a model that identifies tweets potentially associated with the Salafi Jihadist movement in England. The model was trained on a curated dataset of English-language tweets and focuses on identifying the linguistic and pragmatic features indicative of extremist ideologies. The primary goal is to aid in the early detection of extremist content and contribute to enhanced security measures.

## Key Features

- **Multi-Model Approach**: Several machine learning models were developed and evaluated, including:
  - Support Vector Classifier (SVC) with TF-IDF vectorization
  - Deep learning models such as Multi-Layer Perceptron (MLP) and Long Short-Term Memory (LSTM) networks
  - Various data balancing techniques like SMOTE (Synthetic Minority Over-sampling Technique) and Random Under Sampling were also applied to handle class imbalance.
  
- **Ensemble Model**: The final model is an ensemble of the top-performing individual models using a soft voting method. This ensemble approach provided the highest accuracy and robustness in identifying extremist tweets.

- **Data Visualization**: The project includes detailed data visualizations, such as word frequency distributions and class separability graphs, to better understand the characteristics of the dataset.

## Usage

Due to the sensitivity of the dataset, the training data used in this project cannot be shared publicly. However, the repository includes pre-trained models that can be used for evaluation or inference on new data.

- **Pre-trained Models**: Pre-trained models are available in the `models/` folder. You can use these models to perform predictions on your data.

- **Inference**: The `predict.py` script can be used to run predictions on new tweet data. For example:

  ```bash
  python predict.py --input_file sample_input.txt --model models/final_model.pkl
   ```
  
## Results
The final combined model achieved an accuracy of 90.8% and demonstrated high stability in identifying extremist tweets, with a recall of 83% for positive cases in real-world data tests.

## Acknowledgments
This project was supervised by Dr. Eli Alshaikh from the Department of Middle Eastern Studies, Bar-Ilan University. Special thanks to Dr. Roni Ramon for their invaluable support and guidance throughout the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


