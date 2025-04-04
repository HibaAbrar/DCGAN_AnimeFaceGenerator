import torch
from torch import nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import urllib.request  # For downloading model from GitHub
import os
import streamlit as st

# ==== CONFIGURATION ====
MODEL_URL = "https://raw.githubusercontent.com/HibaAbrar/DCGAN_Face_Generator/generator_epoch_150.pth"
MODEL_PATH = "generator_epoch_150.pth"
INPUT_VECTOR_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== DOWNLOAD MODEL IF NOT EXISTS ====
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... This may take a few seconds.")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ==== GENERATOR MODEL (MUST MATCH TRAINING DEFINITION) ====
class Generator(nn.Module):
    def __init__(self, input_vector_dim, feature_map_dim=32, channels=3):
        super(Generator, self).__init__()
        self.convt_1 = nn.ConvTranspose2d(input_vector_dim, feature_map_dim*16, 4, 1, 0, bias=False)  
        self.batch_norm_1 = nn.BatchNorm2d(feature_map_dim*16)

        self.convt_2 = nn.ConvTranspose2d(feature_map_dim*16, feature_map_dim*8, 4, 2, 1, bias=False)  
        self.batch_norm_2 = nn.BatchNorm2d(feature_map_dim*8)

        self.convt_3 = nn.ConvTranspose2d(feature_map_dim*8, feature_map_dim*4, 4, 2, 1, bias=False)  
        self.batch_norm_3 = nn.BatchNorm2d(feature_map_dim*4)

        self.convt_4 = nn.ConvTranspose2d(feature_map_dim*4, feature_map_dim*2, 4, 2, 1, bias=False)  
        self.batch_norm_4 = nn.BatchNorm2d(feature_map_dim*2)

        self.convt_5 = nn.ConvTranspose2d(feature_map_dim*2, channels, 4, 2, 1, bias=False)  

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.batch_norm_1(self.convt_1(x)))
        x = self.relu(self.batch_norm_2(self.convt_2(x)))
        x = self.relu(self.batch_norm_3(self.convt_3(x)))
        x = self.relu(self.batch_norm_4(self.convt_4(x)))
        return self.tanh(self.convt_5(x))

# ==== LOAD TRAINED GENERATOR ====
generator = Generator(INPUT_VECTOR_DIM).to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()

def generate_images():
    noise = torch.randn(1, INPUT_VECTOR_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        generated_image = generator(noise).cpu()
    fig, ax = plt.subplots(figsize=(1, 1), dpi=300) 
    ax.axis("off")
    ax.imshow(np.transpose(vutils.make_grid(generated_image, padding=2, normalize=True), (1, 2, 0)))
    st.pyplot(fig)  

st.title("DCGAN Face Generator")

if st.button("Generate Image"):
    generate_images()
