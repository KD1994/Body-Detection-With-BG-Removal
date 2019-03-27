# Body-Detection-With-BG-Removal
This Web App is used to detect human bodies from the image and then extract the human bodies and apply BG removal technique to get just a human body without any kind of background.

Things used to implement:

1: imageai - Python library developed for object detection. (https://github.com/OlafenwaMoses/ImageAI.git)
I have used YoloV3 for object detection, you can use Retinanet or yolo-tiny as well as mentioned in the documents.

2: Xception Model: used to remove the background of the image (https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)

Make sure to add these models into "static/models/(your_model_folder)" folder to run the main script.
