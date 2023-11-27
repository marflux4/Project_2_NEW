from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the trained model
model = load_model('trained_model')

# Define the classes
classes = ['Large', 'Medium', 'None', 'Small']

# Map class names to indices
class_indices = {class_name: i for i, class_name in enumerate(classes)}

# Function to preprocess and predict the class of an image
def predict_class(image_path):
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)[0]
    
    # Get the actual class
    actual_class = image_path.split('/')[-2]  # Using directory names as classes
    
    # Display the original image with the predicted and actual classes and probabilities
    plt.imshow(img)
    plt.axis('off')
    
    # Display information for each class
    text_actual = f'Actual Class: {actual_class}\n'
    text_predict = "Predicted Classes:\n"
    for i, class_name in enumerate(classes):
        text_predict += f'{class_name}: {predictions[i]:.2%}\n'
    
    plt.text(50, 92, text_actual, color='white', backgroundcolor='black', fontsize=8)
    plt.text(5, 92, text_predict, color='white', backgroundcolor='black', fontsize=8)
    plt.show()

# Test images
test_images = ['Data/Test/Medium/Crack__20180419_06_19_09,915.bmp', 'Data/Test/Large/Crack__20180419_13_29_14,846.bmp']

# Predict classes for each test image
for img_path in test_images:
    predict_class(img_path)