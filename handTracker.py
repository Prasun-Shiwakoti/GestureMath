
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


connectionPoints = [[]]
status = 'Not Writing'
annotatedImage = None
outputText = ''
annotationsScanned = False

def addConnections(coords):
    global connectionPoints
    if len(connectionPoints[-1]) == 0:
        
        connectionPoints[-1].append(coords)
        return
    if (connectionPoints[-1][-1][0] - coords[0])**2 + (connectionPoints[-1][-1][1] - coords[1])**2 > 20:
        connectionPoints[-1].append(coords)

def annotateWritings(image, connectionPoints):
    for connections in connectionPoints:
        image = cv2.polylines(image, [np.array(connections)], False, (0, 255, 0), 2)
    return image

def annotateImage(image, landmarks, color=(0, 255, 0)):
    for landmark in landmarks:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        image = cv2.circle(image, (x, y), 5, color, -1)
    return image

def evaluate_result(result, output_image: mp.Image, timestamp_ms: int):
    global annotatedImage, status
    gestures = result.gestures   
    gesture = 'No Hand'
    if gestures:
        try:
            gesture = gestures[0][0].category_name  
            print(gesture, gestures[0][0].score)
            annotatedImage = annotateImage(output_image.numpy_view().copy(), result.hand_landmarks[0], (0, 255, 0))
            match gesture:
                case 'Closed_Fist':
                    
                    if connectionPoints[-1] != []:
                        connectionPoints.append([])
                    status = 'Not Writing'
                    # print('Close Fist gesture detected')
                case 'Open_Palm':
                    # status = 'Open_Palm'
                    pass
                    # print('Open Palm gesture detected')
                case 'Thumb_Up':
                    status = 'Eraser'
                    pass
                    # print('Thumbs Up gesture detected')
                case 'Thumb_Down':
                    pass
                    # print('Thumbs Down gesture detected')
                case 'Victory':
                    pass
                    # print('Victory gesture detected')
                case 'ILoveYou':
                    exit(1)
                    pass
                    # print('I Love You gesture detected') 
                case 'Pointing_Up':
                
                    if connectionPoints[-1] != []:
                        if gestures[0][0].score > 0.75:
                            status = 'Writing'
                            annotatedImage = annotateImage(annotatedImage, [result.hand_landmarks[0][8]], (255, 0, 0))
                            addConnections([int(result.hand_landmarks[0][8].x * annotatedImage.shape[1]), int(result.hand_landmarks[0][8].y * annotatedImage.shape[0])])
                    else:
                        if gestures[0][0].score > 0.7:
                            status = 'Writing'
                            annotatedImage = annotateImage(annotatedImage, result.hand_landmarks[0], (255, 0, 0))
                            addConnections([int(result.hand_landmarks[0][8].x * annotatedImage.shape[1]), int(result.hand_landmarks[0][8].y * annotatedImage.shape[0])])
                #  
                # print('Pointing Up gesture detected')    
                case _:
                    pass
                    # print('Unknown gesture detected')
        except Exception as e:
            print("Error in evaluate_result function:", e)

    # try:
    #     cv2.imshow('Title', annotatedImage)
    #     if cv2.waitKey(5) & 0xFF == ord("q"):
    #         print("Exiting...")
    #         camera_running = False
    #         exit(1)
    # except Exception as e:
    #     print("Error:", e)
    #     exit(1)

def refresh_screen(annotatedImage, connectionPoints, status, outputText):
    annotatedImage = annotateWritings(annotatedImage, connectionPoints)
    annotatedImage = cv2.putText(annotatedImage, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    annotatedImage = cv2.putText(annotatedImage, outputText, (10, annotatedImage.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return annotatedImage

def clearScreen():
    global connectionPoints
    connectionPoints = [[]]
 

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='models/gesture_recognizer.task'),
    min_tracking_confidence = 0.4,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=evaluate_result
)
recognizer = vision.GestureRecognizer.create_from_options(options)

webcam = cv2.VideoCapture(1)

frame_timestamp = 0

while webcam.isOpened():
    frame_timestamp += 1
    success, img = webcam.read()

    if not success:
        print("Ignoring empty camera frame.")
        break

    img = cv2.resize(img, (640, 480))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if annotatedImage is None:
        annotatedImage = img

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    recognizer.recognize_async(mp_image, frame_timestamp)
    
    annotatedImage = refresh_screen(cv2.cvtColor(annotatedImage, cv2.COLOR_RGB2BGR) , connectionPoints, status, "No Text")

    cv2.imshow('Title', annotatedImage)
    annotatedImage = None
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()

