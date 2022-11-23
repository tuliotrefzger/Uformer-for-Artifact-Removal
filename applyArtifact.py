import os
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.util import random_noise
from PIL import Image
import torchio as tio

## This script applies a certain artifact (degradation) to a specified collection of images.
## In this case, the test set of the brain tumor dataset

#################################### UTILS ###############################################


def to_0255(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    return img


def linear_transform(img, alpha, beta):
    new_img = np.zeros(img.shape, img.dtype)

    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         new_img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)

    # removi o loop que estava causando a lentidão enorme no processamento
    new_img = np.clip(alpha * img + beta, 0, 255)

    return new_img.astype(np.uint8)


## The following artifact functions were made by Gianlucas Lopes

################################ GAUSSIAN NOISE ############################################


def generate_ruido_gaussiano(img, deg_level):
    # Noise Levels
    sigmas = np.linspace(1, 10, 10) / 40
    return to_0255(random_noise(img, var=sigmas[deg_level - 1] ** 2))


################################### CONTRAST ###############################################


def generate_contrast(img, deg_level):
    alpha = (
        11 - deg_level
    ) * 0.09  # Normaliza deg_level em uma escala de 0.09 a 0.90 para ser o alfa da transformação
    beta = (
        128 * (1 - alpha) - 1
    )  # beta é definido de forma a manter o histograma no centro (não gerar imagens apenas mais escuras)

    new_img = linear_transform(
        img, alpha, beta
    )  # Aplica a transformação linear à imagem

    return new_img.astype(np.uint8)


################################### BLURRING ###############################################

# Noise Levels
interval = 1000
sigmasX = np.linspace(1.5, interval, 10)
sigmasY = np.linspace(1.5, interval, 10)
sigmasX = sigmasX / 100
sigmasY = sigmasY / 100

# deg_level de 1 a 10
def generate_blurring(img, deg_level):
    return cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=sigmasX[deg_level - 1],
        sigmaY=sigmasY[deg_level - 1],
        borderType=cv2.BORDER_CONSTANT,
    )


################################## RINGING ##########################################


def create_circular_mask(h, w, r):
    # Mask with center circle as 1, remaining as zeros
    # h,w = image.shape
    # r = circle radius
    ch, cw = h // 2, w // 2  # center coordinates
    y, x = np.ogrid[-ch : h - ch, -cw : w - cw]
    boolmask = x * x + y * y <= r * r
    mask = np.zeros((h, w, 2), np.uint8)
    mask[boolmask] = 1
    return mask


def fourier_apply_mask(im, mask):
    # Converting image to frequency domain
    dft = cv2.dft(np.float64(im), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    fshift = dft_shift * mask  # apply mask
    f_ishift = np.fft.ifftshift(fshift)  # inverse shift
    imfilt = cv2.idft(f_ishift)  # inverse dft
    im_final = cv2.magnitude(
        imfilt[:, :, 0], imfilt[:, :, 1]
    )  # merge real and imaginary parts
    return im_final


# Segunda versão
def generate_ringing(img, deg_level):
    # deg_level = 1 significa pouco ringing e 10 significa muito
    # Normaliza deg_level em uma escala de 50 a 180 para ser o raio do filtro
    radius = np.uint((((deg_level - 11) * -1) - 1) / 9 * 50 + 16)
    print(img.shape)
    h, w = img.shape
    mask = create_circular_mask(h, w, radius)
    img_f = fourier_apply_mask(img, mask)  # img filtrada sem escala com floats
    img0255 = to_0255(img_f)  # img 0255 int
    return img0255


################################## GHOSTING ##########################################


def pil_to_tio(img):
    npimg = np.array(img)
    # A biblioteca torchio precisa dos channels na primeira dimensão
    npimgcwh = np.transpose(npimg, (2, 0, 1))
    # Adicionando mais uma dimensão para ficar 4d
    npimg4d = npimgcwh[..., np.newaxis]
    tioimg = tio.ScalarImage(tensor=npimg4d)
    return tioimg


def tio_to_pil(img4d):
    npimgcwh = np.squeeze(img4d)
    # Voltando o channel para última dimensão
    npimg = np.transpose(npimgcwh, (1, 2, 0))
    npimg255 = to_0255(npimg)
    img = Image.fromarray(npimg255)
    return img


def np_1ch_to_pil(img):
    pimg = np.repeat(img[..., np.newaxis], 3, -1)  # converte para 3 canais
    pimg = Image.fromarray(pimg)
    return pimg


def pil_to_np_1ch(img):
    npimg = np.array(img)
    npimg = np.dot(npimg, [0.299, 0.587, 0.114]).astype(np.uint8)
    return npimg


def generate_ghosting(img, deg_level):
    # deg_level = 1 significa pouco ringing e 10 significa muito
    intensities = np.linspace(0.3, 1.1, 10)
    pil_img = np_1ch_to_pil(img)
    pio_img = pil_to_tio(pil_img)
    func = tio.Ghosting(
        num_ghosts=5, axis=0, intensity=intensities[deg_level - 1], restore=0.02
    )
    new_pio_img = func(pio_img)
    new_pil_img = tio_to_pil(new_pio_img)
    new_np_img = pil_to_np_1ch(new_pil_img)
    return new_np_img


###################################### START #############################################


print("START")
## Choose a degradation to be applied to the dataset
# degradation = "GaussianNoise"
# degradation = "Contrast"
# degradation = "Blurring"
# degradation = "Ringing"
degradation = "Ghosting"

print(("Degradation: " + degradation).upper())
print()

## Directory where the images are located in my local repository (which won't be uploaded to GitHub due to its size).
## If you don't have the dataset with you, it can be found in the commented directory.
directory = "patientImages/splits"
# directory = "/mnt/nas/GianlucasLopes/NeuralBlack/patientImages/splits"

## The destination directory where the degraded images will be stored.
degraded_directory_root = ""

## For loop to generate the 10 different degradation levels
for deg_level in range(1, 11):
    print("Degradation level:", deg_level)
    print()
    if deg_level < 10:
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

    total_images = 0

    ## Walks the files inside the directory "directory".
    for directory_path, subdirectories, files in os.walk(directory):
        for file in files:
            ## Checks if the image is a PNG.
            if file.lower().endswith(".png"):
                total_images += 1
                file_path = directory_path + "/" + file
                degraded_directory_path = (
                    degraded_directory + directory_path[len(directory) :]
                )
                print("image number: ", total_images)

                ## Only apply artifact to images in the test dataset.
                if "test" in directory_path:
                    ## Makes the directories in the degraded test dataset.
                    os.makedirs(name=degraded_directory_path, exist_ok=True)
                    original_file = np.float32(load_img(filepath=file_path))

                    ## Applying the degradation.
                    if degradation == "GaussianNoise":
                        degraded_file = generate_ruido_gaussiano(
                            img=original_file, deg_level=deg_level
                        )
                        degraded_file = cv2.cvtColor(degraded_file, cv2.COLOR_BGR2GRAY)

                    elif degradation == "Contrast":
                        degraded_file = generate_contrast(
                            img=(to_0255(original_file)), deg_level=deg_level
                        )
                        degraded_file = cv2.cvtColor(degraded_file, cv2.COLOR_BGR2GRAY)

                    elif degradation == "Blurring":
                        degraded_file = generate_blurring(
                            img=to_0255(original_file), deg_level=deg_level
                        )
                        degraded_file = cv2.cvtColor(degraded_file, cv2.COLOR_BGR2GRAY)

                    elif degradation == "Ringing":
                        ## Some degradation functions like ringing and ghosting require 2D images, i.e., gray images.
                        original_file = cv2.cvtColor(original_file, cv2.COLOR_BGR2GRAY)
                        degraded_file = generate_ringing(
                            img=original_file, deg_level=deg_level
                        )

                    elif degradation == "Ghosting":
                        original_file = cv2.cvtColor(original_file, cv2.COLOR_BGR2GRAY)
                        degraded_file = generate_ghosting(
                            img=to_0255(original_file), deg_level=deg_level
                        )

                    ## Saving the image in the degraded test dataset.
                    cv2.imwrite((degraded_directory_path + "/" + file), degraded_file)
                    print("Degraded image was created!")
                print()
    print()
    print()


print("END")
