
# Fruit Freshness Detection

This project aims to classify whether a fruit is fresh or rotten using deep learning techniques. It uses Convolutional Neural Networks (CNNs) to identify freshness based on the visual features of fruit images. The model is trained and deployed using a Streamlit web interface for easy interaction.

## Project Description

This is a binary classification project that detects the freshness of fruits like apples, bananas, and oranges. The core of this project is a CNN model built using TensorFlow and Keras, trained on labeled image data. The dataset consists of two categories: fresh and rotten. The system takes an input image and predicts whether the fruit in the image is fresh or rotten.

## Key Components

* `app.py`: Streamlit web application for user interface and prediction
* `fruit_freshness_model.ipynb`: Jupyter Notebook used to preprocess data, build, train, and evaluate the CNN model
* `fruit_classifier_model.h5`: Trained model file (not uploaded here due to size limits)
* `dataset`: Image dataset of fresh and rotten fruits (not included in this repository, available via external link)
* `output/`: Folder containing screenshots of Streamlit app output after deployment and prediction

## Technologies Used

* Python
* TensorFlow and Keras
* OpenCV
* Streamlit
* Jupyter Notebook / Google Colab

## How to Run the Project

1. Clone the repository to your local system.
2. Download the dataset and model file from the provided links and place them in the appropriate folders.
3. Install the required packages using:
   `pip install -r requirements.txt`
4. Run the Streamlit app using:
   `streamlit run app.py`
5. Upload a fruit image in the app to get the freshness prediction.

## Download Links

* Dataset: [https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
* Trained Model (.h5): [https://drive.google.com/drive/folders/1cuQI75eB4-q2ODPXdAOuB9JqN\_P9Z5vr](https://drive.google.com/drive/folders/1cuQI75eB4-q2ODPXdAOuB9JqN_P9Z5vr)

## Folder Structure

fruit_freshness_detection  
├── app.py  
├── fruit_freshness_model.ipynb  
├── fruit_classifier_model.h5 (not uploaded)  
├── dataset (not uploaded)  
├── output/  ← contains Streamlit app output screenshots  


## Author

Riya Dwivedi

