import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Title of the app
st.title("Image Analyzer")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to analyze the colors in the image
def analyze_colors(image):
    st.header("Color Analysis")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Reshape the image to be a list of pixels
    pixels = img.reshape((-1, 3))
    
    # Convert to float type
    pixels = np.float32(pixels)
    
    # Define criteria, number of clusters(K)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, (centers) = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8 bit values
    centers = np.uint8(centers)
    
    # Get the colors
    colors = centers[labels.flatten()]
    
    # Reshape back to the original image
    segmented_image = colors.reshape(img.shape)
    
    # Display the dominant colors
    st.image(segmented_image, caption="Image with Dominant Colors", use_column_width=True)
    
    # Plotting the dominant colors
    st.header("Dominant Colors")
    counts = np.bincount(labels.flatten())
    dominant_colors = centers[np.argsort(-counts)]
    
    fig, ax = plt.subplots()
    for i, color in enumerate(dominant_colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color/255.0))
    ax.set_xlim(0, k)
    ax.set_ylim(0, 1)
    ax.axis("off")
    st.pyplot(fig)

# Function to perform basic face detection using OpenCV
def detect_faces(image):
    st.header("Face Detection")
    img = np.array(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the resulting image
    st.image(img, caption="Image with Detected Faces", use_column_width=True)

# Check if the file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Choose the analysis type
    analysis_type = st.selectbox("Choose an analysis type", ["Color Analysis", "Face Detection"])
    
    if analysis_type == "Color Analysis":
        analyze_colors(image)
    elif analysis_type == "Face Detection":
        detect_faces(image)
