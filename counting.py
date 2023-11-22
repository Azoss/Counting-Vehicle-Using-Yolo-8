import cv2
import pandas as pd
import time
from ultralytics import YOLO
from tracker import Tracker

#-----
model = YOLO('yolov8s.pt')
#-----
def Mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', Mouse)

cap = cv2.VideoCapture('video\\veh6.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

tracker = Tracker()

cy1 = 300
cy2 = 350

slope = 0# Kemiringan garis
offset = 6

vh_down = {}
vh_up = {}

# Definisikan warna garis
line_color = (0, 255, 0)
line_thickness = 2

total_cars = 0
total_trucks = 0
total_buses = 0
total_motorcycles = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list.append([x1, y1, x2, y2])
        elif 'bus' in c:
            list.append([x1, y1, x2, y2])
        elif 'motorcycle' in c:
            list.append([x1, y1, x2, y2])
            
    bbox_id = tracker.update(list)
    updated_ids = []
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

        if cy1 - offset < cy < cy1 + offset:
            vh_down[id] = time.time()
        if id in vh_down and cy2 - offset < cy < cy2 + offset:
            elapsed_time = time.time() - vh_down[id]
            distance = 10  # meters
            a_speed_ms = distance / elapsed_time
            a_speed_kh = a_speed_ms * 3.6
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, str(int(a_speed_kh))+'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            if 'car' in c:
                total_cars += 1
            elif 'truck' in c:
                total_trucks += 1
            elif 'bus' in c:
                total_buses += 1
            elif 'motorcycle' in c:
                total_motorcycles += 1
            del vh_down[id]

        if cy2 - offset < cy < cy2 + offset:
            vh_up[id] = time.time()
        if id in vh_up and cy1 - offset < cy < cy1 + offset:
            elapsed_time = time.time() - vh_up[id]
            distance = 10  # meters
            a_speed_ms = distance / elapsed_time
            a_speed_kh = a_speed_ms * 3.6
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, str(int(a_speed_kh))+'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            
            if 'car' in c:
                total_cars += 1
            elif 'truck' in c:
                total_trucks += 1
            elif 'bus' in c:
                total_buses += 1
            elif 'motorcycle' in c:
                total_motorcycles += 1
            del vh_up[id]
        
        updated_ids.append(id)

    # Gambar garis horizontal
    y1 = int(cy1 + slope * frame.shape[1])
    y2 = int(cy2 + slope * frame.shape[1])
    cv2.line(frame, (0, cy1), (frame.shape[1], y1), line_color, line_thickness)
    cv2.line(frame, (0, cy2), (frame.shape[1], y2), line_color, line_thickness)

    # Tampilkan jumlah mobil yang terhitung
    cv2.putText(frame, "Total Cars: {}".format(total_cars), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Tampilkan jumlah truck yang terhitung
    cv2.putText(frame, "Total Truck: {}".format(total_trucks), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Tampilkan jumlah BUS yang terhitung
    cv2.putText(frame, "Total Bus: {}".format(total_buses), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Tampilkan jumlah motor yang terhitung
    cv2.putText(frame, "Total Motor: {}".format(total_motorcycles), (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()