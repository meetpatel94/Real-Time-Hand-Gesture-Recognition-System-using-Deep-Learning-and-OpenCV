import cv2
import os

path = "dataset/train/05"

files = os.listdir(path)

for i in range(0, len(files), 20):
    img = cv2.imread(os.path.join(path, files[i]))
    cv2.imshow("Sample", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
