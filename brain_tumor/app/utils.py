
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('model/brain_tumor_10_epochs.h5')
def result_interpretation(value):
    if value == 1:
        return "Yes Brain Tumor"
    else:
        return "No Brain Tumor"

def get_result(img):
    image  = cv2.imread(img)
    image = Image.fromarray(image,'RGB')
    image = image.resize((64,64))
    image = np.array(image)
    input_image = np.expand_dims(image,axis=0)
    result = model.predict(input_image)
    predicted_class_index = int(result[0][0])
    return predicted_class_index
