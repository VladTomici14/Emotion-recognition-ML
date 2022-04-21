from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from utils.style import *
import numpy as np
import imutils
import time
import cv2

# ---- loading the models -----
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
            prediction_percentage = int(prob * 100000) / 1000
            text = f"{emotion}: {prediction_percentage}%"

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

    # ------ showing the detected face and the canvas for that face----
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
    # ------------ FPS variables --------------
    previousTime = 0
    timeNow = 0

    camera = cv2.VideoCapture(0)
    while True:
        # ----- reading frame by frame the video from the camera ----
        ret, frame = camera.read()
        if not ret:
            break

        # ------ preprocessing the image before detection -------
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # conversion to gray scale
        faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)  # detecting faces

        # --------- creating the output frames ----------
        frameClone = frame.copy()  # the video output
        canvas = np.zeros((300, 300, 3), dtype="uint8")

        if len(faces) > 0:
            # ----------- drawing the rectangle for the face recog ------
            k = 0
            for (x, y, w, h) in faces:
                k += 1
                perFaceDetection(x, y, w, h, frameClone, emotion_classifier, k)
        else:
            cv2.destroyAllWindows()
            print("I do not see any faces! :(")

        # --------------- calculating the FPS of the video -----------
        timeNow = time.time()
        fps = 1 / (timeNow - previousTime)
        previousTime = timeNow
        fps = str(int(fps))
        cv2.putText(frameClone, fps, (7, 70), font, 1, greenColor, 3, cv2.LINE_AA)

        # ----- showing the image ----
        cv2.imshow("Emotion recognition - output", frameClone)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()