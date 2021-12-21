import cv2

img = cv2.imread('img/face_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = cv2.CascadeClassifier('face.xml')

results = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for(x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

cv2.imshow("Result", img)
cv2.waitKey(0)