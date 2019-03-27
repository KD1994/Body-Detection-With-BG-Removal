from flask import Flask, request, render_template
from imageai.Detection import ObjectDetection
import requests
import os
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import datetime


app = Flask(__name__)
app.config["upload_fd"] = 'static/op_images'
extracted_img_path = []
temp_fname = []
bg_rem_objs_path = []


class DeepLabModel(object):

    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):

        """Creates and loads pretrained deeplab model."""

        self.graph = tf.Graph()

        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):

        """Runs inference on a single image.
            Args:
            image: A PIL.Image object, raw input image.
            Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """

        start = datetime.datetime.now()

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))

        return resized_image, seg_map


def body_detection(file):
    detector = ObjectDetection()

    # set the model type
    # 1. Retina net
    # 2. Yolov3 (currently used)
    # 3. Yolo-tiny
    detector.setModelTypeAsYOLOv3()

    # provide the location of the h5 file
    detector.setModelPath(os.getcwd() + "/static/models/yolov3/yolo.h5")

    # load the model
    detector.loadModel()

    # 80 objects can be detected, but only focusing on Human Bodies
    custom = detector.CustomObjects(person=True)
    detections, extracted_objects = detector.detectCustomObjectsFromImage(custom_objects=custom,
                                                                          input_image=file,
                                                                          output_image_path=os.getcwd() + "/static"
                                                                                                          "/op_images/"
                                                                                                        + file,
                                                                          display_object_name=False,
                                                                          display_percentage_probability=False,
                                                                          extract_detected_objects=True,
                                                                          minimum_percentage_probability=30)
    return detections, extracted_objects


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        url = request.form['URL of the Image']
        # print(url)
        fname = url.split("/")[-1]
        r = requests.get(url, allow_redirects=True)
        open(fname, "wb").write(r.content)
        print(fname)

        if len(temp_fname) == 0:
            temp_fname.append(fname)
        else:
            temp_fname.clear()
            temp_fname.append(fname)

        detections, objects = body_detection(fname)
        # print(path)

        full_fname = os.path.join(app.config["upload_fd"], fname)

        # Get Path for Extracted objects
        for each_obj in objects:
            img_name = each_obj.split("/")[-1]
            extracted_img_path.append(os.path.join(app.config["upload_fd"], fname + "-objects/"
                                                   + img_name))

        return render_template("result.html", detections=detections, user_image=full_fname, objects=extracted_img_path,
                               fname=fname)
    return "Not ok"


@app.route("/extracted", methods=["GET", "POST"])
def extracted():
    modelType = "static/models/xception_model"

    MODEL = DeepLabModel(modelType)
    print('model loaded successfully : ' + modelType.split("/")[-1])

    for each_obj in extracted_img_path:
        img_name = each_obj.split("/")[-1]
        inputFilePath = each_obj
        outputFilePath = os.path.join(app.config["upload_fd"], temp_fname[0] + "-objects/bg_removed_"
                                      + img_name)

        if inputFilePath is None or outputFilePath is None:
            print("Bad parameters. Please specify input file path and output file path")
            exit()

        # Inferences DeepLab model and visualizes result.
        try:
            print("Trying to open : " + inputFilePath.split("/")[-1])
            jpeg_str = open(inputFilePath, "rb").read()
            original_im = Image.open(BytesIO(jpeg_str))
        except IOError:
            print('Cannot retrieve image. Please check file: ' + inputFilePath)
            return

        print('running deeplab on image %s...' % inputFilePath)
        resized_im, seg_map = MODEL.run(original_im)

        width, height = resized_im.size
        dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
        for x in range(width):
            for y in range(height):
                color = seg_map[y, x]
                (r, g, b) = resized_im.getpixel((x, y))
                if color == 0:
                    dummyImg[y, x, 3] = 0
                else:
                    dummyImg[y, x] = [r, g, b, 255]

        img = Image.fromarray(dummyImg)
        img = img.convert("RGB")
        img.save(outputFilePath)
        print("output is stored in %s" % outputFilePath)
        bg_rem_objs_path.append(outputFilePath)

    return render_template("extracted.html", objects=extracted_img_path, bg_rem_objs_path=bg_rem_objs_path)


if __name__ == "__main__":
    app.run()
