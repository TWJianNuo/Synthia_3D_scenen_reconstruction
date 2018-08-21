Last Modified: June 26, 2016

German Ros

info@synthia-dataset.net

http://synthia-dataset.net


SYNTHIA: The SYNTHetic collection of Imagery and Annotations


This package contains the SYNTHIA-SEQS-05-SPRING subset. This is a video stream generated at 5 FPS.
The number of classes here presented covers those defined in the table below, extending the classes

originally defined in our CVPR paper. Instance segmentation groundtruth and global camera poses are provided!


Please, if you use this data for research purposes, consider citing our CVPR paper:


German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, Antonio M. Lopez; Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 3234-3243

@InProceedings{Ros_2016_CVPR,

author = {Ros, German and Sellart, Laura and Materzynska, Joanna and Vazquez, David and Lopez, Antonio M.},

title = {The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

month = {June},

year = {2016}

}


DISCLAIMER: 


The data here presented is part of the full SYNTHIA-SEQS dataset, i.e., video streams. If you look for
the RANDOM sequences it is *NOT* this one. You can find SYNTHIA-RAND in the download section of our website.


DESCRIPTION:

The package contains the following data,

* RGB: 	    	folder containing standard 1280x760 RGB images used for training
* Depth:    	folder containing 1280x760 unsigned short images. Depth is encoded in any of the 3 channels
	    	in centimetres as an ushort
* GT/COLOR:	folder containing png files (one per image). Annotations are given using a color representation.
		This is mainly provided for visualization and you are not supposed to use them for training.
* GT/LABELS:	folder containing png files (one per image). Annotations are given in two channels. The first
		channel contains the class of that pixel (see the table below). The second channel contains
		the unique ID of the instance for those objects that are dynamic (cars, pedestrians, etc.).
* CameraParams:	folder containing text files with global camera poses and intrinsic/extrinsic parameters,
		(useful for visual odometry, 3D reconstruction, etc.)


Class		R	G	B	ID

Void		0 	0 	0	0
Sky             128 	128 	128	1
Building        128 	0 	0	2
Road            128 	64 	128	3
Sidewalk        0 	0 	192	4
Fence           64 	64 	128	5
Vegetation      128 	128 	0	6
Pole            192 	192 	128	7
Car             64 	0 	128	8
Traffic Sign    192 	128 	128	9
Pedestrian      64 	64 	0	10
Bicycle         0 	128 	192	11
Lanemarking	0	172 	0	12
Reserved	- 	- 	-	13
Reserved	- 	- 	-	14
Traffic Light	0 	128 	128	15
