import numpy as np

from tensorflow import keras
import tensorflow.lite as tflite
from keras.utils import load_img

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io


# Application
root_path = "/image_sorting_service/api"
app = FastAPI(
    docs_url=root_path + "/docs",
    openapi_url=root_path + "/v1/openapi.json",
    openapi_tags=[]
)

# Model details 
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
interpreter = tflite.Interpreter(model_path='final_model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def lite_predict_image(image, class_names):
    x = np.array(image)
    X = np.array([x])
    X = np.float32(X)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    pred_class_index = pred.argmax(axis=-1)[0]
    return class_names[pred_class_index]

@app.get(root_path + '/service_health')
def service_health():
    """
    To check if service is up
    """
    return "Service is up"


# Preprocessing function (depends on how your model was trained)
def preprocess_image(image: Image.Image, target_size=(223, 223)) -> np.ndarray:
    image = image.resize(target_size)
    image = image.convert("RGB")
    return image

@app.post(root_path + '/predict_image/')
async def predict_image(file: UploadFile = File(...)):
    """
    Predicts image
    """

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = preprocess_image(image, target_size=(224, 224))
    res = lite_predict_image(image, class_names)
    return f"Prediction: {res}"





 
