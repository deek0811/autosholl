libraries.txt contains all required libraries to use neuron.py. 
Use pip install -r libraries.txt in console to ensure all libraries are installed.

This GitHub Repository contains sample neuron images in the 'data' file. 
The main file, neuron.py, automatically inputs this folder of .tif file images, runs through all images, and outputs into a folder 'results'. 
Folder names can be changed at to accomodate for other images. 
INPUT_DIR = Path("./data")
OUTPUT_DIR = Path("./results")
Output folders are automatically created.

The code will take in an image from the input folder, and output the following in a single folder:
- CSV with information about the branch
- Image of the input in grayscale
- Masked image of the input 
- Overlay image - Green indicates detected sections of the neuron. The red line represents a preliminary skeleton outline.
- Preprocessed image of the input
- Image of the final skeleton mapped
- CSV with information about the neuron
- CSV with Sholl analysis information

All the information can also be found together in the summary CSV found in the main output file.

We can work on this readme, finalize it, copy it over to a new one and delete this one. 


