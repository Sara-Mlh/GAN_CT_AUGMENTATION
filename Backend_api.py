from flask import Flask, jsonify, request
from defs import CTScanDataset, dicom_paths, show_tensor_images, show_generated_images, Generator, Discriminator, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch
import os


app = Flask(__name__)

# Global variables
dataset = None
dataloader = None
generator = None
discriminator = None
G_opt = None  
D_opt = None  
device = "cuda"



def initialize_dataset_and_loader():
    global dataset, dataloader
    #path = "C:/Users/sara/Documents/Notebooks/dataset_idri"
    #path= C:/Users/ASUS/Documents/Notebooks/CT SCAN PF/dataset_idri
    path="C:/Users/ASUS/Documents/Notebooks/CT SCAN PF/manifest-1714049609094/LIDC-IDRI"
    
    file_paths = dicom_paths(path)
    
    dataset = CTScanDataset(file_paths, resize_size=(128, 128))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

def load_gan_model_and_optimizers(checkpoint_dir, epoch):
    global generator, discriminator, G_opt, D_opt
    noise_dim = 100  

    # Initialize models
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    # Initialize optimizers
    G_opt = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    if os.path.isfile(checkpoint_path):
        load_checkpoint(checkpoint_path, generator, discriminator, G_opt, D_opt)
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

# Initialize the dataset and dataloader
initialize_dataset_and_loader()

@app.route("/dataset_info", methods=['GET'])
def dataset_info():
    try:
        num_images = len(dataset)
        num_batches = len(dataloader)
        return jsonify({
            "num_images": num_images,
            "num_batches": num_batches
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/display_tensor", methods=['POST'])
def display_tensor():
    try:
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        my_batch = next(iter(dataloader))
        image_base64 = show_tensor_images(my_batch, num_images=16)
        return jsonify({'image': image_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_images", methods=['POST'])
def generate_images():
    try:
        data = request.json
        epoch = data.get("epoch", 0)  # get the epoch number (or a default value)
        #checkpoint_dir = "C:/Users/sara/Documents/Notebooks/check_gradient"  # Directory for checkpoints
        checkpoint_dir = "C:/Users/ASUS/Documents/Notebooks/check_gradient"

        # Load model and optimizers from checkpoint
        load_gan_model_and_optimizers(checkpoint_dir, epoch)

        # Generate and display images
        image_base64 = show_generated_images(epoch, generator, num_images=16, noise_dim=100)

        if not image_base64:
            return jsonify({"error": "No images generated or encoding failed"}), 500
        
        return jsonify({'image': image_base64})
    except Exception as e:
        print(f"Error in generate_images endpoint: {e}")  # Debugging output
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
