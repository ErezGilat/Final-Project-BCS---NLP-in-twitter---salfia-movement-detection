# NLP Model for Identifying Salafi Jihadist Tweets in England

This repository contains the code and data for the final project titled **"Developing a Natural Language Processing Model to Identify Tweets Associated with the Salafi  Movement in England."** The project was conducted as part of the requirements for a B.Sc degree and is aimed at enhancing the ability to identify and monitor extremist content on social media platforms.

## Project Overview

The project leverages advanced machine learning techniques to develop a model that identifies linguistic and pragmatic features of tweets related to the Salafi Jihadist movement. The model was trained on a diverse and extensive dataset of carefully selected English-language tweets. The primary goal is to assist in the early detection of extremist content and contribute to security measures.

## Key Features

- **Multi-Model Approach**: Several models were developed and tested, including Support Vector Classifiers (SVC) with TF-IDF, deep learning models like MLP (Multi-Layer Perceptron) and LSTM (Long Short-Term Memory), with different sampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) and Random Under Sampling.
- **Ensemble Model**: The final model is a combination of the best-performing individual models using a soft voting ensemble method. This approach provided the highest accuracy and reliability in identifying extremist tweets.
- **Data Visualization**: The project includes visualizations that help in understanding the distribution and characteristics of the data, such as word frequency distributions and class separation
  
## Usage

Due to the sensitivity of the data used in this project, the dataset cannot be shared. As a result, users will not be able to train the models directly. However, the repository includes the pre-trained models, which can be used for evaluation and inference.

- **Pre-trained Models**: The pre-trained models are available in the `models/` folder. You can use these models to evaluate their performance on your own data or test their predictions on sample inputs.

- **Inference**: Use the inference scripts in the `inference/` folder to make predictions on new data. For example:
   ```bash
   python inference/predict.py --input_file sample_input.txt --model models/final_model.pkl
  
## Results
The final combined model achieved an accuracy of 90.8% and demonstrated high stability in identifying extremist tweets, with a recall of 83% for positive cases in real-world data tests.

## Acknowledgments
This project was supervised by Dr. Eli Alshaikh from the Department of Middle Eastern Studies, Bar-Ilan University. Special thanks to Dr. Roni Ramon for their invaluable support and guidance throughout the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


