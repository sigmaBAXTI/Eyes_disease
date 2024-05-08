import streamlit as st
from PIL import Image
import numpy as np
from fastai.vision.all import *
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
koz_rasmlar = os.listdir('Validation')
boshqa_rasmlar = os.listdir('Boshqa')

model = load_learner("Koz_model.pkl")

st.title("Ko'z kasalliklarini tasniflash")
st.write("Iltimos, ko'z kasalligini aniqlash uchun rasm yuklang.")

uploaded_file = st.file_uploader("Rasmni yuklang", type=['png', 'jpg', 'jpeg', 'gif'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.uint8) / 255.0

    
    prediction, _, probs = model.predict(image_array)
    predicted_class = prediction.item()

    if predicted_class in koz_rasmlar:
        st.success(f"Bashorat -> {predicted_class}")
        st.info(f"Ehtimollik  -> {probs[predicted_class] * 100 :.1f} %")
    else:
        st.write("Iltimos, boshqa rasm kiriting!!!")
