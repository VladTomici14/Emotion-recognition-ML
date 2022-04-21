from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from utils.style import *
import numpy as np
import argparse
import imutils
import cv2

# TODO: write the documentation of the entire project
# TODO: explain the concept of argparsing
# TODO: prepare the installs for her

# ---------- argparsing arguments --------
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

# ------ models variables -------
detection_model_path = "models/haarcascade_files/haarcascade_frontalface_default.xml"
emotion_model_path = "models/face_recog_vgg_856.hdf5"

# ----- setting up the classifiers ----
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


def perFaceDetection(x, y, h, w, frameClone, emotionClassifier, number):
    """
    This function will do the actual emotion recognition for a frame.

        :param x: the x coordinate of the detection rectangle
        :param y: the y coordinate of the detection rectangle
        :param h: the height of the detection rectangle
        :param w: the width of the detection rectangle
        :param frameClone: the frame that will be shown
        :param emotionClassifier: the emotion recognition classifier
        :param number: the number of the face that was detected
    """

    # ---- setting up the output and input ----
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    croppedFace = frameClone[y:y + h, x:x + w]

    if type(croppedFace) is np.ndarray:
        # ---- transforming the frame into an array -----
        face = cv2.resize(croppedFace, (64, 64))
        arrayImage = face.astype("float") / 255.0
        arrayImage = img_to_array(arrayImage)
        arrayImage = np.expand_dims(arrayImage, axis=0)

        # --- making the prediction -----
        prediction = emotionClassifier.predict(arrayImage)[0]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, prediction)):
            text = f"{emotion}: {prob * 100}"

            # ---- drawing on the canvas -----
            W = int(prob * 300)
            if EMOTIONS[prediction.argmax()] == emotion:
                cv2.rectangle(canvas, (7, (i * 35) + 5), (W, (i * 35) + 35), greenColor, -1)
            else:
                cv2.rectangle(canvas, (7, (i * 35) + 5), (W, (i * 35) + 35), redColor, -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), font, 0.45, whiteColor, 1)

            # ----- drawing on the output frame ----
            cv2.putText(frameClone, EMOTIONS[prediction.argmax()], (x, y - 10), font, 0.45, blueColor, 2)
            cv2.rectangle(frameClone, (x, y), (x + w, y + h), redColor, 2)

    # ----- showing the detected face and the canvas ------
    croppedFace = cv2.resize(croppedFace, (250, 300))
    result = horizontal_concat_resize([croppedFace, canvas])
    cv2.imshow(f"Face #{number}", result)


def horizontal_concat_resize(img_list):
    """
        This function will concatenate 2 images horizontally.

            :param img_list: this is an array that will contain all the images that will be in the OpenCV format
            :return: will return the concatenated image.

        Basically the principle behind this is to find the image that has the minimum height. We will resize every
        image for that height.

        Why we do this?
        The cv2.hconcat() function simply concatenates images, but only the images that have the same height. We cannot
        simply apply this function because the cropped_face image will always have different sizes.
    """

    h_min = min(img.shape[0] for img in img_list)

    resized_img_list = []
    for img in img_list:
        img = cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=cv2.INTER_CUBIC)
        resized_img_list.append(img)

    # ---- returning the final concatenated image ----
    return cv2.hconcat(resized_img_list)


def main():
    # ---- loading the argparsed image ----
    image = cv2.imread(args["image"])

    # ------ preprocessing the image before detection -------
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    # --------- creating the output frames ----------
    frameClone = image.copy()
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    k = 0
    if len(faces) > 0:
        # ----------- drawing the rectangle for the face recog ------
        for (x, y, w, h) in faces:
            k += 1
            perFaceDetection(x, y, w, h, frameClone, emotion_classifier, k)

    # ----- showing the image ----
    cv2.imshow("Emotion recognition - output", frameClone)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
