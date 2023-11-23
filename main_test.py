import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('brain_tumor_10_epochs.h5')
image = cv2.imread('/Users/q/Desktop/Image Processing/brain_tumor_detection/brain_tumor_dataset/pred/pred5.jpg')
image = Image.fromarray(image)
image = image.resize((64,64))
image = np.array(image)
input_image = np.expand_dims(image,axis=0)
# Predict probabilities for each class
probs = model.predict(input_image)
print(probs)

predicted_class_index = int(probs[0][0])

print(f"Predicted class: {predicted_class_index}")