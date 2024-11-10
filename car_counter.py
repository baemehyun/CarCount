from ultralytics import YOLO
import numpy as np
import cv2
import math
from sort import *  # noqa: F403

# cap = cv2.VideoCapture('output.mp4') #for video file
cap = cv2.VideoCapture('output1-full.mp4') #for video file

model = YOLO('yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
# classNames = [ "car" ]

mask = cv2.imread('mask3.jpg')

# Create SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # noqa: F405

limits = [535, 346, 1495, 552]
# limits = [183, 183, 664, 362]
totalCounts = []
currentframe = 0
while True:
    sucess, img = cap.read()
    if not sucess: 
        break
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detection = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #confidence value
            conf = math.ceil((box.conf[0]*100))/100            
            
            #class name
            classIndex = int(box.cls[0])
            print("classIndex",classIndex)
            if classIndex < len(classNames) :
                current_class = classNames[classIndex]
            else:
                print("when??")
                pass

            if current_class == 'car' and conf > 0.9:
                name = './data-1/detected-frame' + str(currentframe) + '.jpg'
                cv2.imwrite(name, img)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                current_array = np.array([[x1, y1, x2, y2, conf]])
                name = './data-detect-1/detected-frame' + str(currentframe) + '.jpg'
                print ('Creating...' + name) 
        
                # writing the extracted images 
                cv2.imwrite(name, img) 
        
                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1

                detection = np.vstack((detection, current_array))
                print("detection",detection)
    
    # Update tracker
    print("update tracker")
    resultsTracker = tracker.update(detection)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id  = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1

        cv2.putText(img, f'car {id}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
            if id not in totalCounts:
                totalCounts.append(id)
                # print("line")
                cv2.line(img, (cx, cy), (cx, cy), (255, 0, 0), 1)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 4)
    cv2.putText(img, f' Count : {len(totalCounts)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
