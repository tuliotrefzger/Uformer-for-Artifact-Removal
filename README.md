# Uformer for MRI Artifact Removal


## Ojective

This project is part of my graduation thesis whose objective was to test whether the application of [Uformer](https://github.com/ZhendongWang6/Uformer) was capable of mitigating the negative effects that MRI artifacts (Gaussian noise, Contrast, Blurring, Ringing, and Ghosting) have on the accuracy of a neural network designed to classify three types of brain tumor (meningiomas, gliomas, and pituitary tumors).


## Dataset

The dataset used in this project is the [Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427). In case you are working with the _Portainer_ service from the GPDS Reasearch group from the University of Brasília (UnB), the path to the dataset is already put into the files as "/mnt/nas/GianlucasLopes/NeuralBlack/patientImages/splits". In case you are not in this research group or for some reason the directory is no longer available, download the dataset and add its path where it is needed in the code.


## Instructions

* If for some reason you cannot access the dataset through the "/mnt/nas/GianlucasLopes/NeuralBlack/patientImages/splits" path or you do not have access to the GPDS GPU server, download the dataset from the link above and replace the line containing this old path for whatever path you decide to give to this dataset.
* Run applyArtifact.py to produce the 10 different degraded versions of the dataset corresponding to the 10 levels of degradation. Repeat this process for the 5 different artifacts.
* Run applyUformerDirectory.py to produce the 10 restored versions of the degraded artifacts. Repeat this process for the 5 different artifacts.
* Go to [brain-tumor-classifier-with-Uformer](https://github.com/tuliotrefzger/brain-tumor-classifier-with-Uformer) and follow the next steps.
