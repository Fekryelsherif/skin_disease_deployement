import streamlit as st
from PIL import Image
import numpy as np
from keras import models



def load_image(image_file):
    img = Image.open(image_file)
    return img


st.title("skin diseases classification")

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
model = models.load_model('expskin.h5')

if image_file is not None:
    st.image(load_image(image_file), width=250)
    image = Image.open(image_file)
    image = image.resize((200, 200))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 200, 200, 3)
    result = model.predict(image_arr)
    ind = np.argmax(result)
    classes = ['Allergy', 'Bacteria']
    print(ind)
    st.header(classes[ind])
    