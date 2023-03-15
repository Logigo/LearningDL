import math
from typing import Tuple

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# The image can come in any size, but we want to resize it to be consistent so we can use the same patch number during training


# Create our dataloader
flowers_data = torchvision.datasets.Flowers102('./', split="train", download=True)
data_loader = torch.utils.data.DataLoader(flowers_data,
                                          batch_size=4,
                                          shuffle=True)

number_of_patches = 16
patch_size = 512 // number_of_patches

def turn_image_to_patch(img: torch.Tensor, patch_size: int):
    """
    Step 1.1:
    Reshapes img from (H x W x C) to a sequence of patches of N x (P*P*C)
    Where N is the number of patches, or H / P

    This can be made faster if images were passed in batches (batch_size x H x W x C),
    but will be omitted for demonstration purposes. This can also be combined with the transformation pipeline above.
    We will do this later.
    """
    C, H, W = img.shape

    assert H % patch_size == 0 and W % patch_size == 0, \
     f'Image dimensions must each be whole multiples of the patch size! Received ' + \
     f'{H=}{W=}{C=} and {patch_size=}'

    # From the docs of unfold: "Currently, only 4-D input tensors (batched image-like tensors) are supported."
    # Since we have a single image, we need to 'unsqueeze' it, which adds an extra axis (3, 512, 512) --> (1, 3, 512, 512) to make it a batch of size 1.
    # Also, because we want patches with no overlap, we set the stride to be the same as the patch size.
    # This means that after overlapping every patch of size kernel_size, we will have moved the window by patch_size pixels.
    patches = torch.nn.functional.unfold(img.unsqueeze(0), kernel_size=patch_size, stride=patch_size)
    print(f'Image reshaped from {H=} {W=} {C=} to {patches.shape}')
    return patches

# Testing with a single image:
image, label = flowers_data[0]
patches = turn_image_to_patch(image, patch_size)


def positional_encoding(tensor: torch.Tensor, embedding_dimension: int) -> torch.Tensor:
    """
    Step 1.4:
    Returns a positional encoding for the given tensor.

    This uses a mixture of sine and cosine functions to encode the position of each patch.
    In particular, it makes use of their frequency components to encode the position of each patch.
    We are going to combine the positional encoding with the patches, so we will need to make sure that the positional encoding
    has the same dimension as the patches, so we pass in the embedding_dimension.
    """
    length = tensor.size(0)
    # This holds the encoding
    encoding = torch.zeros(length, embedding_dimension)
    # This holds the position of each patch
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    # This will be used to calculate the frequency of each sine/cosine function
    # e^([0, 2, 4, ...embedding_dimension] * -log(10000) / embedding_dimension)
    division_term = torch.exp(torch.arange(0, embedding_dimension, 2) * (-math.log(10000.0) / embedding_dimension))
    # [0, 2, 4, ...embedding_dimension] will be element wise multiplied by position, and then sin is applied
    encoding[:, 0::2] = torch.sin(position * division_term)
    # [1, 3, 5, ...embedding_dimension] will be element wise multiplied by position, and then cos is applied
    encoding[:, 1::2] = torch.cos(position * division_term)
    return encoding

class VisionTransformer(nn.Module):
    def __init__(self, image_size: Tuple[int, int], number_of_patches: int, num_classes: int, embedding_dimension: int=768) -> None:
        super().__init__()
        h, w = image_size
        self.reshape_image = transforms.Compose([
            transforms.Resize((h, w)), # Resize knows to keep the channel dimension
            transforms.ToTensor(),
        ])

        self.number_of_patches: int = number_of_patches
        self.patch_size: int = h // number_of_patches
        patch_pixels = self.patch_size * self.patch_size
        # Embed the patches to a vector of 768 to learn lower-dimensional representations
        self.embedding_dimension = embedding_dimension
        self.patch_embedding = nn.Linear(patch_pixels * 3, embedding_dimension)

        # Base transformer object which we will use to build the encoder, which is a stack of 12 transformer layers.
        self.transformer = nn.TransformerEncoderLayer(d_model=768, nhead=12)
        self.encoder_layers = nn.TransformerEncoder(self.transformer, num_layers=12)

        # MLP 'head' to 'learn' from the classification token
        self.mlp_head = nn.Sequential([
            nn.Linear(embedding_dimension, 256), # 256 is the mlp's hidden dimension. Feel free to parameterize this.
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        ])


    def forward(self, image):
        # Step 1.1: Building patches from an image
        reshaped = self.reshape_image(image)
        patches = turn_image_to_patch(reshaped, self.patch_size)

        # Step 1.2: Projecting the patches
        # Linear embedding:
        embedded_patches = self.patch_embedding(patches)

        # Step 1.3: Appending a 'classification token' to the patches.
        # Notice how it is the same size as 1 patch.
        classification_token = torch.zeros(1, self.embedding_dimension)
        # Each patch is a row, and we want another 'patch' as a classification token. So we append a row of zeros, alongside the row dimension.
        embedded_patches = torch.cat([embedded_patches, classification_token], dim=0)

        # Step 1.4: Building a "positional encoding" for each patch
        # This will be the same size as the patches, as it has to be concatenated to the patches.
        positionally_encoded_patches = positional_encoding(embedded_patches, self.embedding_dimension)


        # Step 1.5: Combining the positional encodings with the patches via concatenation
        combined_patches =  torch.cat([positionally_encoded_patches, embedded_patches], dim=1)

        # TODO:
        # Steps 2: Transformer blocks!
        # We will use the TransformerEncoder from PyTorch, which is a stack of TransformerEncoderLayers.
        encoded_output = self.encoder_layers(combined_patches)

        # Step 3: Use the MLP to learn from the classification token
        # Because we added the classification token to the end, we can just take the last row
        encoded_classification_token = encoded_output[-1]
        # We can now pass this through the MLP head
        logits = self.mlp_head(encoded_classification_token)
        # We have the final output to learn from!
        return logits


    def __call__(self, image):
        return self.forward(image)
