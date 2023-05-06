from typing import List, Optional

import torch
from torch.nn import DataParallel

from models.eyenet import EyeNet
import os
import numpy as np
import cv2
import dlib
import util.gaze
import copy
import math
from imutils import face_utils
from PIL import ImageGrab
# import keyboard
import sys,tty

from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample
    
torch.backends.cudnn.enabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


dirname = os.path.dirname(__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(dirname, 'lbpcascade_frontalface_improved.xml'))
landmarks_detector = dlib.shape_predictor(os.path.join(dirname, 'shape_predictor_5_face_landmarks.dat'))

checkpoint = torch.load('checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'])


webcam_width = 960
webcam_height = 480
textColor = (255, 38, 233)
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
webcam.set(cv2.CAP_PROP_FPS, 60)

# get screen size so the window can be full screened 
screen = ImageGrab.grab().size
screen_width = math.floor(screen[0]/2)
screen_height = math.floor(screen[1]/2)
# rad = 50
# circles = [
#     (rad, rad, rad, 'Circle 1'),
#     (math.floor(webcam_width/2) - rad, rad, rad, 'Circle 2'),
#     (rad, webcam_height - rad, rad, 'Circle 3'),
#     (math.floor(webcam_width/2) - rad, webcam_height - rad, rad, 'Circle 4')
# ]

# circles = [(50, 50, 30), (100, 50, 30), (100, 100, 30), (50, 100, 30)]  # List to hold the coordinates of the circles
# next_circle = 0  # Index of the circle to add or delete next

# def mouse_click(event, x, y, flags, param):
#     global next_circle

#     if event == cv2.EVENT_LBUTTONDOWN:
#         if next_circle < len(circles):
#             if ((x - circles[next_circle][0])**2 + (y - circles[next_circle][1])**2) < circles[next_circle][2]**2:  # Check if the click is inside the circle
#                 circles.pop(next_circle)  # Remove the clicked circle from the list
#             else:
#                 next_circle += 1  # Move to the next circle to add or delete


# detect when user clicks on window, set it to focus mode
# def on_mouse(event, x, y, flags, param):
#     print("here")
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("left button clicked")
#         # cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# # detect when user clicks circle
# def on_mouse(event, x, y, flags, param):
#     print("on_mouse_circle")
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("on_mouse_circle: click detected")
#         if (x - 100)**2 + (y - 100)**2 <= 100**2:
#             print("Circle clicked!")

# detect when user clicks circle
# def on_mouse(event, x, y, flags, param):
#     print("here")
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("left button clicked")
#         for circle in circles:
#             center_x, center_y, radius, label = circle
#             if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
#                 print(label + ' clicked!')

# circle_coords = [(50, 50), (100, 50), (100, 100), (50, 100)]
# radius = 20
# color = (0, 255, 0)
# thickness = 2
# circles = []
# current_circle = None

# def key_callback(event, x, y, flags, params):
#     global current_circle, circles
    
#     # If space bar is pressed
#     if event == cv2.EVENT_KEYDOWN and event == ord(' '):
#         if current_circle is not None:
#             # Delete the current circle
#             circles.remove(current_circle)
#             current_circle = None
            
#         # Add the next circle
#         if len(circles) < len(circle_coords):
#             next_circle = circles[len(circles)]
#             circles.append(cv2.circle(img, next_circle, radius, color, thickness))
#             current_circle = circles[-1]
            
#         cv2.imshow("image", img)

rad = 50
circles = [(rad, rad), (screen_width - rad, rad), (rad, screen_height - rad*2), (screen_width - rad, screen_height - rad*2)]
current_circle = 0

def draw_circle(img, circle):
    print("draw_circle")
    print("circle:", circle)
    print("current_circle:", current_circle)
    print("circles:", circles)
    cv2.circle(img, circle, rad, (0, 255, 0), -1)
    cv2.putText(img, str(circles.index(circle)), (circle[0]-5, circle[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def key_callback(event, x, y, flags, param):
    global current_circle, circles
    
    if event == cv2.EVENT_LBUTTONUP:
        for circle in circles:
            if np.sqrt((circle[0]-x)**2 + (circle[1]-y)**2) < 20:
                print("Circle clicked:", circles.index(circle))
                # delete_circle(circle)
                break

    elif event == cv2.EVENT_FLAG_ALTKEY:
        cv2.destroyAllWindows()

    elif event == cv2.EVENT_FLAG_CTRLKEY:
        cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elif event == cv2.EVENT_FLAG_SHIFTKEY:
        if current_circle < len(circles):
            current_circle += 1

    elif event == cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_CTRLKEY:
        cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elif event == cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_SHIFTKEY:
        # delete_circle(circles[current_circle])
        if current_circle < len(circles):
            current_circle += 1

def main():
    global current_circle, circles
    calibrationComplete = False
    # next_circle = 0
    # Create a full screen window with the full screen button enabled
    # cv2.namedWindow('Webcam', cv2.WINDOW_FULLSCREEN)

    # cv2.setMouseCallback('Webcam', mouse_click) 
    cv2.namedWindow("Webcam")
    cv2.setMouseCallback("Webcam", key_callback)
    cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    gaze_set = False
    current_face = None
    landmarks = None
    alpha = 0.95
    left_eye = None
    right_eye = None
    white = (255, 255, 255)
    black = (0, 0, 0)

    while True:
        _, frame_bgr = webcam.read()
        orig_frame = frame_bgr.copy()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        if len(faces):
            next_face = faces[0]
            if current_face is not None:
                current_face = alpha * next_face + (1 - alpha) * current_face
            else:
                current_face = next_face

        if current_face is not None:
            next_landmarks = detect_landmarks(current_face, gray)

            if landmarks is not None:
                landmarks = next_landmarks * alpha + (1 - alpha) * landmarks
            else:
                landmarks = next_landmarks

        gaze_left = []
        gaze_right = []
        gaze_left_txt = ''
        gaze_right_txt = ''

        if landmarks is not None:
            eye_samples = segment_eyes(gray, landmarks)

            eye_preds = run_eyenet(eye_samples)
            left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
            right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

            if left_eyes:
                left_eye = smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
            if right_eyes:
                right_eye = smooth_eye_landmarks(right_eyes[0], right_eye, smoothing=0.1)

            for ep in [left_eye, right_eye]:
                for (x, y) in ep.landmarks[16:33]:
                    # green circle for right
                    color = (0, 255, 0)
                    # blue circle for left
                    if ep.eye_sample.is_left:
                        color = (255, 0, 0)
                    cv2.circle(orig_frame,
                               (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
                gaze = ep.gaze.copy()
                if ep.eye_sample.is_left:
                    gaze[1] = -gaze[1]
                    gaze_left = copy.deepcopy(gaze)
                else:
                    gaze_right = copy.deepcopy(gaze)
                util.gaze.draw_gaze(orig_frame, ep.landmarks[-2], gaze, length=60.0, thickness=2)
            
                
        # put gaze indexes on the screen
        cv2.putText(orig_frame, 'Gaze Green:' + str(gaze_left), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2, cv2.LINE_AA)
        cv2.putText(orig_frame, 'Gaze Blue:' + str(gaze_right), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2, cv2.LINE_AA)
        
        # create a rectangle frame around the window to indicate wether or not user is looking at a safe zone
        frameColor = black
        if current_circle >= len(circles):
            frameColor = gaze_zone(gaze_left, gaze_right)
        cv2.rectangle(orig_frame, (5,5), (1275, 715), frameColor, 5)
        
        
        # add interface to detect the users eye gaze zone when looking at their screen
        if current_circle < len(circles):
            cv2.putText(orig_frame, 'Look at the green circle and press SPACE.', (math.floor(webcam_width/4)-100, math.floor(webcam_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, black, 2, cv2.LINE_AA)
        # cv2.circle(orig_frame, (50,50), 50, frameColor, -1)

        # cv2.circle(orig_frame, (100, 100), 100, (255, 0, 0), -1)
        # cv2.putText(orig_frame, 'Click me!', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # for circle in circles:
        #     center_x, center_y, radius, label = circle
        #     cv2.circle(orig_frame, (center_x, center_y), radius, (255, 0, 0), -1)
        #     cv2.putText(orig_frame, label, (center_x , center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        # draw circle
        # for i in range(next_circle):
        #     cv2.circle(orig_frame, (circles[i][0], circles[i][1]), circles[i][2], (0, 255, 0), -1)  # Draw circles on the image

        # display window in the screen size
        orig_frame = cv2.resize(orig_frame, (screen_width, screen_height - 50), interpolation = cv2.INTER_AREA)
        
        if current_circle < len(circles):
            draw_circle(orig_frame, circles[current_circle])

        # display window
        cv2.imshow("Webcam", orig_frame)  

        cv2.setMouseCallback("Webcam", key_callback)

        key = cv2.waitKey(1)

        if key & 0xFF == ord(' '):
            # delete_circle(circles[current_circle])
            if current_circle < len(circles):
                current_circle += 1
        
        if key & 0xFF == ord('q'):
            break
        # cv2.setMouseCallback('Webcam', on_mouse) 

        # Set the window property to be always on top
        # Set a mouse callback function to handle mouse events
        # cv2.setMouseCallback('frame', mouse_callback)

        # Check if the user pressed the 'space' key (ASCII code 32)
        key = cv2.waitKey(1)
        # print("key", key)
        # cv2.putText(orig_frame, 'key:' + str(key), (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2, cv2.LINE_AA)
        
        # if key == ord('q'):
        #     break
        # elif key == 32:
        #     if next_circle < len(circles):
        #         next_circle += 1  # Move to the next circle to add or delete
        # elif key == ord('d'):
        #     if next_circle > 0:
        #         next_circle -= 1  # Move to the previous circle to add or delete
        #         circles.pop(next_circle)  # Remove the current circle from the list

        # elif key == 32:
        #     print('space pressed')
        # else: 
            # print('pressed: ', key)
        #     cv2.putText(orig_frame, 'space button pressed', (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2, cv2.LINE_AA)
    # Release the video capture device and close the OpenCV window
    webcam.release()
    cv2.destroyAllWindows()

        # if (gaze_set == False): 
        #     print("entered gaze set")
        #     # tty.setcbreak(sys.stdin)     
        #     for cc in circle_coordinates:
        #         cv2.circle(orig_frame, cc, 50, white, -1)
        #         # read when a user hits the space key
        #         key = ord(sys.stdin.read(1))  # key captures the key-code 
        #         if key == 32:
        #             print("you pressed space")
        #             cv2.putText(orig_frame, 'Key pressed', (800, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2, cv2.LINE_AA)
        #             # register what the persons gaze is
        #     gaze_set = True  



# ***********************************************
# ***********************************************
# New created functions
# ***********************************************
# ***********************************************

# determine the eye zone (red, yellow, green) based on the eye gaze directions
def gaze_zone(left_pupil, right_pupil):
    # set x and y coordinates of each pupil
    left_x, left_y, right_x, right_y = 0, 0, 0, 0

    if (len(left_pupil) >= 2):
        left_y = left_pupil[0]
        left_x = left_pupil[1]
    
    if (len(right_pupil) >= 2):
        right_y = right_pupil[0]
        right_x = right_pupil[1]

    
    # zone colors in RGB
    red = (50, 50, 255)
    yellow = (15, 215, 255)
    green = (71, 252, 80)

    # margin or error for the green and yellow zone
    margin = 0.05

    # ranges of x-coordinate
    # left pupil (blue)
    l_x_max = 0.3 # shouldn't be larger - no need to keep track of right eye for x_max
    # right pupil (green)
    r_x_min = -0.3 # shouldn't be smaller - no need to keep track of left eye x_min

    # ranges of y-coordinate
    y_min = 0 # shouldnt be smaller
    y_max = 0.3 # shouldnt be larger


    # if red zone - looking at screen
    if ((left_x <= l_x_max) and                 #far left of screen
        (right_x >= r_x_min) and                #far right of screen
        (left_y >= y_min or right_y >= y_min)   #far top of screen - no need to keep track of bottom of screen because we want to make sure user is looking upwards not down at phone
        ):
        return red
    
    # if yellow zone - somewhere in between the screen and off
    elif ((l_x_max < left_x <= (l_x_max + margin)) or                 #far left of screen
          (r_x_min > right_x >= (r_x_min - margin)) or                #far left of screen
          ((y_min > left_y >= (y_min - margin)) or 
           (y_min > right_y >= (y_min - margin))) #far top of screen
          ):
        return yellow
    
    # off screen and safe
    else:
        return green
        




# ***********************************************
# ***********************************************
# Previously existing functions
# ***********************************************
# ***********************************************

def detect_landmarks(face, frame, scale_x=0, scale_y=0):
    (x, y, w, h) = (int(e) for e in face)
    rectangle = dlib.rectangle(x, y, x + w, y + h)
    face_landmarks = landmarks_detector(frame, rectangle)
    return face_utils.shape_to_np(face_landmarks)


def draw_cascade_face(face, frame):
    (x, y, w, h) = (int(e) for e in face)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def draw_landmarks(landmarks, frame):
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)


def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

        if is_left:
            eye_image = np.fliplr(eye_image)
            # cv2.imshow('left eye image', eye_image)
        # else:
        #     cv2.imshow('right eye image', eye_image)
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes


def smooth_eye_landmarks(eye: EyePrediction, prev_eye: Optional[EyePrediction], smoothing=0.2, gaze_smoothing=0.4):
    if prev_eye is None:
        return eye
    return EyePrediction(
        eye_sample=eye.eye_sample,
        landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks,
        gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze)


def run_eyenet(eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img], dtype=torch.float32).to(device)
            _, landmarks, gaze = eyenet.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            gaze = np.asarray(gaze.cpu().numpy()[0])
            assert gaze.shape == (2,)
            assert landmarks.shape == (34, 2)

            landmarks = landmarks * np.array([oh/48, ow/80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            assert landmarks.shape == (34, 3)
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            assert landmarks.shape == (34, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
    return result
  
if __name__ == '__main__':
    main()