import os
import sys
import cv2
from datetime import datetime

if not os.path.isfile(f'./videos/{sys.argv[1]}'):
    os.exit()
vid = cv2.VideoCapture(f'./videos/{sys.argv[1]}')
timestamp = datetime.today().strftime('%Y%m%d%H%M%S')

if not os.path.isdir(./gesture_data):
    os.mkdir(f'./gesture_data')
os.mkdir(f'./gesture_data/{timestamp}')
os.mkdir(f'./dataset/{timestamp}/left')
os.mkdir(f'./dataset/{timestamp}/right')

f = open(f'./splits/gesture/{sys.argv[2]}_files.txt',"a")

frame_i = 0
while True:
    ret, frame = vid.read()

    if frame is None:
        break

    width = int(frame.shape[1]/2)

    imgL = frame[:,:width,:]
    imgR = frame[:,width:,:]

    cv2.imwrite(f'./dataset/{timestamp}/left/{frame_i}.png',imgL)
    f.write(f'{timestamp}/left {frame_i} l\n')
    cv2.imwrite(f'./dataset/{timestamp}/right/{frame_i}.png',imgR)
    f.write(f'{timestamp}/right {frame_i} r\n')
    frame_i += 1

f.close()


