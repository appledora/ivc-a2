import cv2
import numpy as np
from utils import *


def find_frame_difference(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Frame Difference D(x,y,t) = |I(x,y,t) - I(x,y,t-1)|
       NOTE: Setting t-1=0 gives the bakground difference assuming the first frame is devoid of the object of interest
    Args:
        current (np.ndarray): I(x,y,t)
        previous (np.ndarray): I(x,y,t-1)

    Returns:
        np.ndarray: D(x,y,t)
    """
    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(current, previous)
    return frame_diff

def my_motion_energy(mh):
    """
    Function that accumulates the frame differences for a certain number of pairs of frames
    Args:
        mh Vector of frame difference images
    Returns:
        dst The destination grayscale image to store the accumulation of the frame difference images
    """
    dst = np.zeros(np.shape(mh[0]), dtype=np.uint8)
    mask = np.logical_or.reduce([m == 255 for m in mh])
    dst[mask] = 255
    return dst

def find_contours(frame):
    """
    Find object contours in the frame difference
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(frame, 127, 255, 0, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_projection_bounding_box(frame):
    """Function to generate bounding boxes from horizontal and vertical projection profiles 

    Args:
        frame (_type_): input image/frame

    Returns:
        np.ndarray: image with bounding box
    """
    _, mask, _, _, _ = find_skin_color_blobs(frame)
    mask = preprocess(mask) # Morphological operation

    horiz = np.sum(mask, axis=1) #horizontal projection
    vert = np.sum(mask, axis=0) #vertical projection
    
    # bounding coordinates
    # Y - coordinates from horizontal projection profile
    y = np.argmax(horiz>0) #first occurence of non-zero pixel value
    yh = horiz.shape[0] - np.argmax(horiz[::-1]>0) - 1 #last occurence of non-zero pixel value

    # Y - coordinates from horizontal projection profile
    x = np.argmax(vert>0) # first occurence of non-zero pixel value
    xw = vert.shape[0] - np.argmax(vert[::-1]>0) - 1 #last occurence of non-zero pixel value

    cv2.rectangle(mask, (x, y), (xw, yh), (255, 0, 0), 2)  # Draw bounding box
    return mask, horiz, vert
    
def convert_to_direction(orientation):
    """
    Convert orientation angle to human-readable direction format
    """
    directions = ["East", "Southeast", "South", "Southwest", "West", "Northwest", "North", "Northeast"]
    angle_deg = np.degrees(orientation) % 360
    index = round(angle_deg / 45) % 8
    return directions[index]

def find_skin_color_blobs(frame):
    """
    Find skin color blobs in the frame
    Source: https://stackoverflow.com/questions/8753833/exact-skin-color-hsv-range
    """
    frame = frame.copy()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get the mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Draw bounding boxes around the contours
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box
    return frame, mask, largest_contour, x, y

def get_shape_position_orientation(shape_frame, contour, x, y):
    """
    Get the position and orientation of the shape
    Ref: https://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter2.pdf
    """

    moment = cv2.moments(contour)
    centroid_x = int(moment["m10"] / moment["m00"])
    centroid_y = int(moment["m01"] / moment["m00"])
    centroid = (centroid_x, centroid_y)

    # Fit an ellipse to the contour
    shape = cv2.fitEllipse(contour) #TODO: Could also use polygon
    # Get the orientation of the object (major axis angle)
    orientation = np.deg2rad(shape[2])
    # alternative: using moments
    a = moment["m20"] - centroid_x*moment["m10"]
    b = 2 * (moment["m11"] - centroid_y*moment["m10"])
    c = moment["m02"] - centroid_y*moment["m01"]
    sin_2theta = b/np.sqrt(b**2 + (a-c)**2)
    cos_2theta = (a-c)/np.sqrt(b**2 + (a-c)**2)
    theta = 1/2 * np.arctan(b/(a-c))
    E_min = (a+c)/2 - (a-c)/2 * cos_2theta - b/2 * sin_2theta
    E_max = (a+c)/2 + (a-c)/2 * cos_2theta + b/2 * sin_2theta
    circularity = E_min/E_max

    # Get the position of the object
    position = shape[0]

    # Get area of the object
    size = moment["m00"]

    length = 100 #TODO: Adjust length
    # x2 = int(centroid_x + length * np.cos(orientation))
    # y2 = int(centroid_y + length * np.sin(orientation))
    # x2 = int(centroid_x + length * cos_2theta)
    # y2 = int(centroid_y + length * sin_2theta)
    # orientation_vector = (x2, y2)
    # orientation_text = convert_to_direction(orientation) #TODO: needs more work
    # Minor Axis
    dx_minor = int(length * np.cos(theta + np.radians(90)))
    dy_minor = int(length * np.sin(theta + np.radians(90)))
    cv2.line(shape_frame, (centroid_x - dx_minor, centroid_y - dy_minor), (centroid_x + dx_minor, centroid_y + dy_minor), 
             (0, 255, 0), 2)
    cv2.ellipse(shape_frame, shape, (0, 0, 255), 2)
    cv2.putText(shape_frame, f"Area: {size}, Orientation: {90 + np.rad2deg(theta)}, Circularity: {circularity}", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return shape_frame

def get_pyramid(frame: np.ndarray, levels=6):
    """Generate Gaussian pyramids
       Ref: https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html

    Args:
        frame (np.ndarray): input image/frame
        levels (int, optional): number of downsamples. Defaults to 6.
    """
    src=frame
    pyramid = [src]
    for _ in range(levels-1):
        src = cv2.pyrDown(src)
        pyramid.append(src)
        
    return pyramid

