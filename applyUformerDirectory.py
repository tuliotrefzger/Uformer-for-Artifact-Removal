import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from model import Uformer

## This scrip applies Uformer to remove a certain degradation (artifact) from a specified
## collection of images. In this case, the test set of the brain tumor dataset.


#################################### UTILS ###############################################
def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    return img


def to_0255(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[
        :, :, ((X - h) // 2) : ((X - h) // 2 + h), ((X - w) // 2) : ((X - w) // 2 + w)
    ] = timg
    mask[
        :, :, ((X - h) // 2) : ((X - h) // 2 + h), ((X - w) // 2) : ((X - w) // 2 + w)
    ].fill_(1.0)

    return img, mask


def return2originalSize(square_img, original_height, original_width):
    _, _, square_size, _ = square_img.size()
    #     print("square_size: ", square_size)
    horizontal_fill = square_size - original_width
    #     print("horizontal_fill: ", horizontal_fill)
    vertical_fill = square_size - original_height
    #     print("vertical_fill: ", vertical_fill)

    if horizontal_fill % 2:
        padding_left = int((horizontal_fill - 1) / 2)
    else:
        padding_left = int(horizontal_fill / 2)

    if vertical_fill % 2:
        padding_top = int((vertical_fill - 1) / 2)
    else:
        padding_top = int(vertical_fill / 2)

    return square_img[
        :,
        :,
        padding_top : (original_height + padding_top),
        padding_left : (original_width + padding_left),
    ]


##########################################################################################

print(f"START\n")

## Choose the GPU intended for training.
if torch.cuda.is_available():
    # device_name = "cuda" # colab
    device_name = "cuda:0"  # servidor
else:
    device_name = "cpu"

device = torch.device(device_name)
print(f"Training in: {device_name}\n")

## Create a blank Uformer model.
model = Uformer(embed_dim=16, token_mlp="leff", img_size=128, use_checkpoint=True)
model = torch.nn.DataParallel(model)

## Load the trained weight and biases learned from training.
FILE = "model5.pth"
model.load_state_dict(torch.load(FILE)["state_dict"])
model.to(device)

## Choose the degradation to be ameliorated
degradation = "GaussianNoise"
# degradation = "Contrast"
# degradation = "Blurring"
# degradation = "Ringing"
# degradation = "Ghosting"

print(("Degradation: " + degradation).upper())
print()

## Directory where the unmodified dataset images are located.
unmodified_dataset_directory = (
    "/mnt/nas/GianlucasLopes/NeuralBlack/patientImages/splits"
)

## Directory containing the degraded images (same as in the ApplyArtifact.py file)
degraded_directory_root = ""

## The destination directory where the restored images will be stored.
restored_directory_root = ""

## Loop that applies Uformer to the 11 different degradation levels (0 to 10).
for deg_level in range(0, 11):
    print("Degradation level:", deg_level)
    print()
    ## Its a good ideal to test if the application of Uformer in undegraded images (i.e., deg_level=0)
    ## increases performance.
    if deg_level == 0:
        degraded_directory = unmodified_dataset_directory
    elif deg_level < 10:
        degraded_directory = (
            degraded_directory_root
            + "patientImages/"
            + degradation
            + "/splits"
            + degradation
            + "0"
            + str(deg_level)
        )
    else:
        degraded_directory = (
            degraded_directory_root
            + "patientImages/"
            + degradation
            + "/splits"
            + degradation
            + str(deg_level)
        )
    if deg_level < 10:
        restored_directory = (
            restored_directory_root
            + "patientImages/"
            + degradation
            + "/splits"
            + "AdjustedUformer"
            + "0"
            + str(deg_level)
        )
    else:
        restored_directory = (
            restored_directory_root
            + "patientImages/"
            + degradation
            + "/splitsAdjustedUformer"
            + str(deg_level)
        )

    total_images = 0

    ## Walks the files inside the directory "degraded_directory".
    for directory_path, subdirectories, files in os.walk(degraded_directory):
        for file in files:
            ## Checks if the image is a PNG.
            if file.lower().endswith(".png"):
                total_images += 1
                file_path = directory_path + "/" + file
                print("Image number ", total_images, file_path)
                restored_directory_path = (
                    restored_directory + directory_path[len(degraded_directory) :]
                )
                # print("Restored directory:", restored_directory_path)

                ## Only apply artifact to images in the test dataset.
                if "test" in directory_path:
                    ## Makes the directories in the restored test dataset.
                    os.makedirs(name=restored_directory_path, exist_ok=True)

                    ## Loads each degraded image.
                    degraded_image = torch.from_numpy(
                        np.float32(load_img(filepath=file_path))
                    )

                    ## Numpy image arrays are of shape [H (height), W (width), C (channel)], but pyTorch works with the
                    ## shape [C, H, W].
                    degraded_image = degraded_image.permute(2, 0, 1)
                    # degraded_image = torchvision.transforms.Resize(256)(degraded_image)

                    ## Getting the height and width of each image (they are all 512x512 in this dataset, but I wanted to be generic).
                    _, original_height, original_width = degraded_image.shape

                    ## Emptying the cache memory.
                    torch.cuda.empty_cache()
                    ## Sets the model to the GPU.
                    model.cuda()
                    ## Sets evaluation mode.
                    model.eval()

                    ## Sets the images to GPU.
                    degraded_image = degraded_image.cuda()
                    ## Unsqueezing the dimension zero from each image (shape [1, C, H, W])
                    degraded_image = degraded_image.unsqueeze(0)
                    ## Uformer works with square images where the sides are a multiple of 128.
                    degraded_image, mask = expand2square(
                        timg=degraded_image, factor=128
                    )

                    ## Emptying the cache memory.
                    torch.cuda.empty_cache()

                    ## Applies the degraded image to Uformer.
                    restored_image = model(degraded_image)
                    ## Returns the image to its original dimensions.
                    restored_image = return2originalSize(
                        restored_image, original_height, original_width
                    )
                    ## Returns the image to the Numpy format.
                    restored_image *= 255
                    restored_image = (
                        restored_image.squeeze(0)
                        .detach()
                        .cpu()
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    ## Makes the image gray.
                    restored_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2GRAY)

                    ## Saves the image in memory.
                    cv2.imwrite(
                        (restored_directory_path + "/" + file),
                        restored_image,
                    )
                    print("Image restored!")
                print()
    print()
    print()

print("END")
