import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

cap = cv2.VideoCapture(0)
cap.set(10, 150)

prototxtPath = r"Face-Mask-Detection-master\face_detector\deploy.prototxt"
weightsPath = r"Face-Mask-Detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

maskNet = tf.keras.models.load_model('face_mask.model')

# Face detection ############################

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")/255
		preds = maskNet.predict(faces.reshape(-1, 224, 224, 3)).ravel()

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
#####################################################
# Test image
# img = cv2.imread('HarryPotter.jpg')
#
# (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
#
# print('locs: ', locs)
# print('\n')
# print('preds: ', preds)
######################################################
# Real-time camera

while True:
    _, frame = cap.read()

    # frameCanny = cv2.Canny(frame, 100, 100)
    # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    # eyes = eye_cascade.detectMultiScale(grey, 1.3, 5)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)
        text = 'Mask {}%'.format(np.round((1-pred)*100, 1)) if pred < 0.5 else 'Non-Mask {}%'.format(np.round(pred*100, 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=4)
        cv2.putText(frame, text=text, org=(x1, y1-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=color, thickness=1)


    # for x, y, w, h in faces:
    #     face_detected = frame[y:y+h, x:x+h,:]
    #     face_detected = cv2.cvtColor(face_detected, cv2.COLOR_BGR2RGB)
    #     face_detected = cv2.resize(face_detected, (150, 150))
    #     # cv2.imshow('face_detected', face_detected)
    #     pred = model.predict(face_detected.reshape(-1, 150, 150, 3))[0, 0]
    #     if pred >= 0.5:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    #         cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 0, 255), -1)
    #         cv2.putText(frame, 'NON-MASK', (x+3, y-7), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
    #     else:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #         cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 255, 0), -1)
    #         cv2.putText(frame, 'MASK', (x + 3, y - 7), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)




    #for e_x, e_y, e_w, e_h in eyes:
        # cv2.rectangle(frame, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 255, 0), 3)



    # cv2.imshow('Frame Canny', frameCanny)
    cv2.imshow('frame', frame)
    # cv2.imshow('grey', grey)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

###################################
# #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
#
# model = tf.keras.models.load_model('face_mask.model')
#
#
# while True:
#     _, frame = cap.read()
#
#     frameCanny = cv2.Canny(frame, 100, 100)
#
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     #faces = face_cascade.detectMultiScale(grey, 1.3, 5)
#     eyes = eye_cascade.detectMultiScale(grey, 1.3, 5)
#
#
#
#     for x, y, w, h in eyes:
#         x_new = x-3*w
#         y_new = y-3*w
#         eye_detected = frame[y_new-3*w:y_new+9*h, x_new-3*w:x_new+8*h,:]
#         cv2.imshow("detect", eye_detected)
#         eye_detected = cv2.resize(eye_detected, (150, 150))
#         # cv2.imshow('face_detected', face_detected)
#         pred = model.predict(eye_detected.reshape(-1, 150, 150, 3))[0, 0]
#         if pred < 0.5:
#             cv2.rectangle(frame, (x_new, y_new), (x_new + 6*w, y_new + 7*h), (0, 0, 255), 3)
#             cv2.rectangle(frame, (x_new, y_new - 25), (x_new + 6*w, y_new), (0, 0, 255), -1)
#             cv2.putText(frame, 'NON-MASK', (x_new+3, y_new-7), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
#         else:
#             cv2.rectangle(frame, (x_new, y_new), (x_new + 6*w, y_new + 7*h), (0, 255, 0), 3)
#             cv2.rectangle(frame, (x_new, y_new - 25), (x_new + 6*w, y_new), (0, 255, 0), -1)
#             cv2.putText(frame, 'MASK', (x_new + 3, y_new - 7), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
#
#
#
#
#     #for e_x, e_y, e_w, e_h in eyes:
#         # cv2.rectangle(frame, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 255, 0), 3)
#
#
#
#     # cv2.imshow('Frame Canny', frameCanny)
#     cv2.imshow('frame', frame)
#     # cv2.imshow('grey', grey)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()