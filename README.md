# CNN-Project
Technion deep learning project in CNNs.

Introduction:
An object proposal technique to identify potential landmarks within an image for place recognition. We use the astonishing power of convolutional neural network features to identify matching landmark proposals between images to perform place recognition over extreme appearance and viewpoint variations. Our implementation is based on the workflow presented in "Place Recognition with ConvNet Landmarks" article, including small modifications and additional experiments to test the ability to perform place recognition technique using tiny objects/boxes (Using EdgeBoxes and SIFT). The whole code is written using python.

Link to Original paper:
https://nikosuenderhauf.github.io/assets/papers/rss15_placeRec.pdf

How to run:
1) In the edgeB/Sift file, make sure to change the maximum area of boxes according to what you need. You can also limit the number of boxes that you want to generate.
2) In matcher file, make sure you receive the model.yml.gz file as an input arg for the program (sys.argv), also change the filesd and filesn variables according to the path of the datasets that you're willing to use for your test (filesd for the dataset that you're willing to iterate over and search for matchable pictures from query images, filesn is for the query images). You can also change the Gaussian random projection parameter in this file. (512/1024/4096...)
3) Run matcher.py after changing the dir path in the filesd and filesn.

List of packages needed to run the project:
1) TorchVision 0.1.8
2) pillow 7.0.0
3) numpy 1.17.3
4) opencv (cv2) 4.1.1
5) scipy 1.1.0
6) scikit-learn 0.22.2
