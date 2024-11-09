import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64


st.title("Data Augmentation of CT scans")
st.write("Department of Artificial intelligence")
st.write("By: MOUALHI Sarah")
st.write("Here is the test application for generating CT Scan images of pulmonary nodules")


tabs = st.tabs(["Home", "Results", "Evaluation"])

with tabs[0]:
    st.header("Home")
    st.write("Welcome to the Home tab.")
    st.write("In this section, we will be exploring our dataset which is a subset of the famous LIDC-IDRI dataset for CT scan images of pulmonary nodules ")
    
    # Fetch dataset information from the Flask backend
    try:
        response = requests.get("http://127.0.0.1:8080/dataset_info")
        if response.status_code == 200:
            #st.write("Response from backend:", response.json())  # Display the raw JSON response
            
            data_info = response.json()
            num_images = data_info.get("num_images", "N/A")
            num_batches = data_info.get("num_batches", "N/A")
            
            st.write(f"Number of images in the dataset: {num_images}")
            st.write(f"Number of batches: {num_batches}")
        else:
            st.error(f"Failed to fetch dataset information. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

    st.write("You can click this button to display a tensor of the dataset's images")
    original_tensor_bt = st.button("Show tensor")

    if original_tensor_bt:
        # Send a POST request to the Flask backend
        response = requests.post("http://127.0.0.1:8080/display_tensor")
        
        if response.status_code == 200:
            # Decode the base64-encoded image
            img_data = base64.b64decode(response.json()['image'])
            img = Image.open(BytesIO(img_data))
            
            # Display the image in Streamlit
            st.image(img, caption="Batch of CT Scan Images", use_column_width=True)
        else:
            st.error("Failed to load images")

with tabs[1]:
    st.header("Results")
    st.write("This is the Results tab.")
    st.write("In this section, we will display a tensor of the generated images of our model per some epochs to show the progress and process of generating images.")
    
    
    max_epochs = 100 
    selected_epoch = st.slider("Select an epoch", min_value=0, max_value=max_epochs, value=0, step=1)
    
    st.write(f"You selected epoch {selected_epoch}:")

   
    generate_images_bt = st.button("Generate Images")

    if generate_images_bt:
        # send a POST request to the Flask backend with the selected epoch
        response = requests.post("http://127.0.0.1:8080/generate_images", json={"epoch": selected_epoch})
        
        if response.status_code == 200:
            # Decode the base64-encoded image
            img_data = base64.b64decode(response.json()['image'])
            img = Image.open(BytesIO(img_data))
            
           
            st.image(img, caption=f"Generated Images from Epoch {selected_epoch}", use_column_width=True)
        else:
            st.error("Failed to load images")

with tabs[2]:
    st.header("Evaluation")
    st.write("This is the Interpretations tab.")
    #image_path = "path/to/your/image.png"  

     
    image = Image.open('loss_plot_fr_transparent.png')

   
    st.image(image, use_column_width=True)
    image2 =  Image.open('metrics_transparent.png')
    st.image(image2, use_column_width=True)







