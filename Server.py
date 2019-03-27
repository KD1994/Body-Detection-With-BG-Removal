from flask import Flask, request, render_template
from imageai.Detection import ObjectDetection
import requests
import os

# from flask_restful import Resource, Api

app = Flask(__name__)
app.config["upload_fd"] = os.path.join("static", "op_images")


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
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom,
                                                       input_image=file,
                                                       output_image_path=os.getcwd() + "/static/op_images/" + file,
                                                       display_object_name=False,
                                                       display_percentage_probability=False,
                                                       extract_detected_objects=False,
                                                       minimum_percentage_probability=30)
    # file = os.getcwd() + "/static/op_images/" + file
    return detections


# api = Api(app)

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
        # print(fname)
        detections = body_detection(fname)
        # print(path)

        full_fname = os.path.join(app.config["upload_fd"], fname)

        return render_template("result.html", detections=detections, user_image=full_fname)
    return "ok"


if __name__ == "__main__":
    app.run(debug=True)
