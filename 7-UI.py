# Import essential libraries 
import streamlit as st 
import tensorflow as tf 
import random 
from PIL import Image, ImageOps
import numpy as np
import cv2 
from cv2 import cvtColor, COLOR_BGR2RGB

# Hide deprication warnings 
import warnings
warnings.filterwarnings("ignore")

# Set configurations for the page 
st.set_page_config(
    page_title = "SkinAI", 
    initial_sidebar_state = 'auto'
)

# Code hiding 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
# CSS hiding, allow Streamlit to run safely as HTML 
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
    st.title("SkinAI")
    st.subheader("Skin disease detection")


model = tf.keras.models.load_model("model.h5", compile = False)
img_path = None
capture_pressed = st.button("Capture Photo")
cap = cv2.VideoCapture(0)
if capture_pressed: 
    ret, frame = cap.read()
    if ret: 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB", caption = "Captured photo")
        img_path = 'your_skin.jpg'
        cv2.imwrite(img_path, frame)
        st.write("Photo saved as:", img_path)
    else: st.write("Failed to capture photo")
cap.release()
cv2.destroyAllWindows() 
    
file = img_path
if (file is None): 
    st.write("Failed to capture photo")
else: 
    img = Image.open(file)
    img1 = img.resize((224, 224))
    iArray = tf.keras.preprocessing.image.img_to_array(img1)
    iArray = tf.expand_dims(iArray, 0)
    p = model.predict(iArray)
    class_names = ['Acne and Rosacea Photos', 'Light Diseases and Disorders of Pigmentation', 'Melanoma Skin Cancer Nevi and Moles'] 
    st.text(class_names[np.argmax(p)]) 
   



