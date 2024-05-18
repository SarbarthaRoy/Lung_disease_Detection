# Lung Disease Diagnosis from X-Ray images built using Keras and CNN

This is a lung disease diagnosis program implemented using Keras and Convolutional Neural Networks (CNN) model. The program is designed to classify chest X-ray images into three categories: COVID-19, Pneumonia, and Normal.

## Dataset
The dataset used to train the model contains chest X-ray images of patients with COVID-19, Pneumonia, and normal cases. The images are divided into two categories - disease (COVID-19 and Pneumonia) and normal cases. This dataset is utilized to train the model to accurately classify chest X-ray images into these categories.

Link to the dataset : https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

## Files

1. `lung_disease_diagnosis.py`: This Python file contains the code for building and training the CNN model on the chest X-ray images dataset. It utilizes the Keras library for implementing the model.

2. `main.py`: This is the main Python file where the trained model is loaded from the `model.h5` file. It provides functionality for users to input a chest X-ray image, and the model will predict the disease category (COVID-19, Pneumonia, or Normal) based on the input image.

3. `model.h5`: This is the pre-trained model file in the Hierarchical Data Format (HDF5). You can use this file to load the trained model in your program.

## Usage

1. Ensure you have the required libraries installed by running the following command:
   ```
   pip install tensorflow keras numpy matplotlib
   ```

2. Training the Model (Optional):
   - If you have your own dataset and wish to train the model, you can customize the code in the `lung_disease_diagnosis.py` file as per your dataset and requirements.
   - Run the `lung_disease_diagnosis.py` file to train the model on your dataset. Adjust the parameters such as batch size, number of epochs, etc., for better accuracy and results.

3. Using the Trained Model:
   - To predict the disease category from an input chest X-ray image, run the `main.py` file.
   - The program will prompt you to provide the path to the input image.
   - After providing the image path, the model will predict the disease category for the given input image.

Feel free to use this code as a starting point and fine-tune it based on your dataset and requirements to achieve more accurate results.

**Note:** Please ensure that the input chest X-ray images are in a compatible format (e.g., PNG or JPEG) and have the appropriate dimensions as expected by the model.

If you encounter any issues or have suggestions for improvements, feel free to raise an issue or contribute to the project!

**Note:** The requirements.txt may contain additional packages if you have any other dependencies in your environment. Make sure to remove any unnecessary packages or versions before sharing the file. Also, it's always a good practice to keep your dependencies updated. If you want to install the exact versions specified in the requirements.txt file in a different environment, you can use the following command:

pip install -r requirements.txt
