a = [
    [
        [384, 162], [381, 154], [382, 146], [385, 141], [391, 141], [396, 137], [403, 136], [410, 139], [415, 140], [423, 145], [430, 154], [434, 161], [436, 172], [433, 181], [430, 189], [429, 194], [423, 201], [414, 205], [405, 208], [400, 207], [395, 209], [390, 214], [394, 209], [399, 206], [407, 203], [416, 200], [423, 200], [428, 201], [433, 201], [449, 214], [450, 220], [451, 227], [453, 232], [453, 237], [452, 247], [452, 252], [451, 257], [449, 262], [441, 273], [434, 276], [429, 276], [419, 274], [413, 270], [408, 267], [403, 263], [398, 260], [394, 257], [398, 262]
    ]
]
def smoothenLines(connectionPoints, window_size=5):
    # Implement a smoothing algorithm (e.g., moving average or Kalman filter) to reduce the jitters.
    smoothed_points = []
    
    for vertices in connectionPoints:
        # Convert vertices to a numpy array for easier manipulation
        points = np.array(vertices)
        
        # Create an array to hold the smoothed points
        smoothed = np.copy(points)
        
        # Apply moving average along the x and y axis separately
        for i in range(1, len(points)):
            start_index = max(0, i - window_size + 1)
            smoothed[i] = np.mean(points[start_index:i+1], axis=0)  # Take the mean of last 'window_size' points
        
        smoothed_points.append(smoothed)
    
    return smoothed_points


import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'Tesseract ocr\tesseract.exe'
import cv2

def detect_handwriting_tesseract(image):
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/@#"'  # PSM 6 is assuming single block of text
    text = pytesseract.image_to_string(image, config=custom_config)

    return text


import numpy as np

image = np.ones((640, 480), dtype=np.uint8) * 255

for connections in smoothenLines(a):
        image = cv2.polylines(image, [np.array(connections)], False, (0, 0, 0), 2)

detected_text = detect_handwriting_tesseract(image)
