import numpy as np
import tensorflow as tf
import cv2
import os
import math
import pickle
from PIL import Image
from preprocess import cnn_model_fn
from selenium import webdriver

glob_prediction = -1
with open(r"./train_data/train_dictionary",'rb') as fp:
    dictionary = pickle.load(fp)

print(dictionary)

def doaction():
    driver = webdriver.Chrome("./chromedriver.exe")
    driver.get("https://google.co.in")
    print("Enter y  to quit: ")
    while input()!='y': pass
    driver.quit()
def nothing(x):
    pass

def identify_gestures(hand_image,classifier):
    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(hand_image)
    img_w, img_h = img.size
    M = max(img_w, img_h)

    background = Image.new('RGB', (M, M), (0, 0, 0))
    bg_w, bg_h = background.size
    offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))

    background.paste(img, offset)
    size = 128,128
    background = background.resize(size, Image.ANTIALIAS)
    open_cv_image = np.array(background)
    background = open_cv_image.astype('float32')
    background = background / 255
    background = background.reshape((1,) + background.shape)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": background},
        num_epochs=1,
        shuffle=False)
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    print(dictionary[predicted_classes[0]],"\nIf the input detected is correct press y else press n:")
    inp = str(input())
    if inp == 'y':
        doaction()
    else:
        pass
    return dictionary[predicted_classes[0]]
    

def get_cam_input(classifier):
    cv2.namedWindow('Camera Output',cv2.WINDOW_GUI_NORMAL)

    # Get pointer to video frames from primary device
    videoFrame = cv2.VideoCapture(0)

    # Process the video frames
    keyPressed = -1  # -1 indicates no key pressed. Can press any key to exit

    # cascade xml file for detecting palm. Haar classifier
    palm_cascade = cv2.CascadeClassifier('palm.xml')

    # previous values of cropped variable
    x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0

    # previous cropped frame if we need to compare histograms of previous image with this to see the change.
    # Not used but may need later.
    _, prevHandImage = videoFrame.read()

    # previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
    prevcnt = np.array([], dtype=np.int32)

    # gesture static increments when gesture doesn't change till it reaches 10 (frames) and then resets to 0.
    # gesture detected is set to 10 when gesture static reaches 10."Gesture Detected is displayed for next
    # 10 frames till gestureDetected decrements to 0.
    gestureStatic = 0
    gestureDetected = 0

    while keyPressed < 0:  # any key pressed has a value >= 0

        # Getting min and max colors for skin
        min_YCrCb = np.array([0,130,103], np.uint8)
        max_YCrCb = np.array([255,182,130], np.uint8)

        # Grab video frame, Decode it and return next video frame
        readSucsess, sourceImage = videoFrame.read()

        # Convert image to YCrCb
        imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
        imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)

        # Find region with skin tone in YCrCb image
        skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

        # Do contour detection on skin region
        _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # sorting contours by area. Largest area first.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # get largest contour and compare with largest contour from previous frame.
        # set previous contour to this one after comparison.
        cnt = contours[0]
        ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
        prevcnt = contours[0]

        # once we get contour, extract it without background into a new window called handTrainImage
        stencil = np.zeros(sourceImage.shape).astype(sourceImage.dtype)
        color = [255, 255, 255]
        cv2.fillPoly(stencil, [cnt], color)
        handTrainImage = cv2.bitwise_and(sourceImage, stencil)

        # if comparison returns a high value (shapes are different), start gestureStatic over. Else increment it.
        if (ret > 0.70):
            gestureStatic = 0
        else:
            gestureStatic += 1

        # crop coordinates for hand.
        x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

        # place a rectange around the hand.
        cv2.rectangle(sourceImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

        # if the crop area has changed drastically form previous frame, update it.
        if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
                    abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
            x_crop_prev = x_crop
            y_crop_prev = y_crop
            h_crop_prev = h_crop
            w_crop_prev = w_crop

        # create crop image
        handImage = sourceImage.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                    max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

        # Training image with black background
        handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
                         max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]

        # if gesture is static for 10 frames, set gestureDetected to 10 and display "gesture detected"
        # on screen for 10 frames.
        if gestureStatic == 10:
            gestureDetected = 10;
            print("Gesture Detected")
            letterDetected = str(identify_gestures(handTrainImage,classifier))

        if gestureDetected > 0:
            if (letterDetected != None):
                pass
            gestureDetected -= 1

        gray = cv2.cvtColor(handImage, cv2.COLOR_BGR2HSV)
        palm = palm_cascade.detectMultiScale(gray)
        for (x, y, w, h) in palm:
            cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = sourceImage[y:y + h, x:x + w]

        # to show convex hull in the image
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        # counting defects in convex hull. To find center of palm. Center is average of defect points.
        count_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            if count_defects == 0:
                center_of_palm = far
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1
                if count_defects < 5:
                    center_of_palm = (far[0] + center_of_palm[0]) / 2, (far[1] + center_of_palm[1]) / 2
            cv2.line(sourceImage, start, end, [0, 255, 0], 2)

        # drawing the largest contour
        cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)

        # Display the source image and cropped image
        cv2.imshow('Camera Output', sourceImage)

        # Check for user input to close program
        keyPressed = cv2.waitKey(30)  # wait 30 miliseconds in each iteration of while loop
        # Close window and camera after exiting the while loop
    cv2.destroyWindow('Camera Output')
    videoFrame.release()


def main(unsed_argv):
    # Create Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model")
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

    #Get Cmera Input and Predict the gesture
    get_cam_input(classifier)

if __name__=="__main__":
    tf.app.run()
