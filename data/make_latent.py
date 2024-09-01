import os
import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm
import argparse

# Function to expand a grayscale image to RGB by repeating the channels
def expand_grayscale_to_rgb(x):
    return x.repeat(3, 1, 1)

# Main function to process the images
def main(input_folder, output_folder, device):
    # Load the AutoencoderKL model and move it to the specified device
    ae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Define the transformation pipeline for the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(expand_grayscale_to_rgb),
    ])

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the list of image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image file
    for image_file in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0).to(device)
        encoded_tensor = ae_model.encode(img).latent_dist.mode().squeeze(0).cpu().detach()
        
        torch.save(encoded_tensor, os.path.join(output_folder, image_file + '.pt'))
    
    print("Done!")

# Entry point for the script
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process images and extract latent representations using an Autoencoder.")
    parser.add_argument('--input_folder', type=str, default='./Footprint', help="Path to the folder containing input images.")
    parser.add_argument('--output_folder', type=str, default='./Footprint_latent', help="Path to the folder where output latent tensors will be saved.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the model on. Default is 'cuda:0'.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.input_folder, args.output_folder, args.device)
