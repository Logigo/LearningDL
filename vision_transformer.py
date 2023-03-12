from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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


class VisionTransformer(nn.Module):
    def __init__(self, image_size: Tuple[int, int], number_of_patches: int) -> None:
        super().__init__()
        h, w = image_size
        self.reshape_image = transforms.Compose([
            transforms.Resize((h, w)), # Resize knows to keep the channel dimension
            transforms.ToTensor(),
        ])

        self.number_of_patches: int = number_of_patches
        self.patch_size: int = h // number_of_patches


    def forward(self, image):
        # Step 1.1: Building patches from an image
        reshaped = self.reshape_image(image)
        patches = turn_image_to_patch(reshaped, self.patch_size)
        # TODO:
        # Step 1.2: Projecting the patches


        # TODO:
        # Step 1.3: Appending a 'classification token' to the patches

        # TODO:
        # Step 1.4: Building a "positional encoding" for each patch

        # TODO:
        # Step 1.5: Combining the positional encodings with the patches

        # TODO:
        # Steps 2: Transformer blocks!
