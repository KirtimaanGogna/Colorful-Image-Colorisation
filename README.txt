# Colorful Image Colorization

*   Abhijoy Sarkar 2019AIM1001
*   Kirtimaan Gogna 2019AIM1014
*   Deepankar Adhikari 2019CSM1004

The file "report.ipynb" consists of the Jupyter notebook report.

The folders consists of relevant files to run the code.
The CIFAR-100 model can be run in the Jupyter Notebook itself or an IPython console while the Zhang et. al's model
needs to be run within an IPython console only.
Implementation instructions are provided in the report itself.

cifar_colorization.py			--->	Runs colorization on CIFAR-100 dataset
						Requires CIFAR-100 dataset stored in "drive/My Drive/cifar-100-python"
						to run. (Dataset not provided)
bw-colorization/images			--->	Store image to be colorized
bw-colorization/video			--->	Store video to be colorized
bw-colorization/bw2color_image.py	--->	Runs Zhang's colorization on images
bw-colorization/bw2color_video.py	--->	Runs Zhang's colorization on video clip or webcam depending on arguments