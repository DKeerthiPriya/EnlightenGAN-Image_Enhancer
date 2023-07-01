from enlighten_inference import EnlightenOnnxModel
import cv2

img = cv2.imread('truck.jpg')
model = EnlightenOnnxModel()

processed = model.predict(img)

cv2.imwrite("truck_result.jpg", processed)