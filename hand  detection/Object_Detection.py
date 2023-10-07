import cv2
import numpy as np
import time
net = cv2.dnn.readNet('training_last.weights', 'testing.cfg') # Belirtilen dosya adına göre yapılandırmayı ve çerçeveyi otomatik olarak algılamasını saglar

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines() # The splitlines() method splits a string into a list

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3)) #  verilen aralık içindeki herhangi bir değerin tek tip tarafından çizilmesi eşit derecede olasıdır.

prev_frame_time = 0
new_frame_time = 0

frame_skip = 20
frame_counter = 0
while True:
    _, img = cap.read()

    frame_counter += 1
    if frame_counter / frame_skip == 1:
        frame_counter = 0
        continue
    #print(img.shape)
    height, width, _ = img.shape


    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]

            rec = cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0,0,0), 2)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    #print(fps)

    cv2.putText(img, "Fps: " + fps, (50, 50), font, 1, (0, 0, 0), 2)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()