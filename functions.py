import cv2
import numpy as np
from utils import *
import os 
import glob
import matplotlib.pyplot as plt

def find_frame_difference(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Frame Difference D(x,y,t) = |I(x,y,t) - I(x,y,t-1)|
       NOTE: Setting t-1=0 gives the bakground difference assuming the first frame is devoid of the object of interest
    Args:
        current (np.ndarray): I(x,y,t)
        previous (np.ndarray): I(x,y,t-1)

    Returns:
        np.ndarray: D(x,y,t)
    """
    
    frame_diff = cv2.absdiff(current, previous)
    gs = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    frame_diff_mask = (gs > 50).astype(np.uint8) * 255
    masked_frame_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=frame_diff_mask)
    return frame_diff_mask

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
    lower_skin = np.array([0, 80, 90], dtype=np.uint8)
    upper_skin = np.array([20, 230, 240], dtype=np.uint8)

    # Threshold the HSV image to get the mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, mask, None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Draw bounding boxes around the contours
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box
    return frame, mask, largest_contour, x, y


def degree_to_plaintext(degree):
    if degree > 337.5 or degree <= 22.5:
        return "East"
    elif 22.5 < degree <= 67.5:
        return "Northeast"
    elif 67.5 < degree <= 112.5:
        return "North"
    elif 112.5 < degree <= 157.5:
        return "Northwest"
    elif 157.5 < degree <= 202.5:
        return "West"
    elif 202.5 < degree <= 247.5:
        return "Southwest"
    elif 247.5 < degree <= 292.5:
        return "South"
    elif 292.5 < degree <= 337.5:
        return "Southeast"

def get_orientation_features(moment):
    """
    Get the orientation of the object
    """
    centroid_x = int(moment["m10"] / moment["m00"])
    centroid_y = int(moment["m01"] / moment["m00"])
    a = moment["m20"] - centroid_x*moment["m10"]
    b = 2 * (moment["m11"] - centroid_y*moment["m10"])
    c = moment["m02"] - centroid_y*moment["m01"]
    sin_2theta = b/np.sqrt(b**2 + (a-c)**2)
    cos_2theta = (a-c)/np.sqrt(b**2 + (a-c)**2)
    theta = 1/2 * np.arctan(b/(a-c))
    # convert radian to degree
    E_min = (a+c)/2 - (a-c)/2 * cos_2theta - b/2 * sin_2theta
    E_max = (a+c)/2 + (a-c)/2 * cos_2theta + b/2 * sin_2theta
    circularity = E_min/E_max
    return (centroid_x, centroid_y), theta, circularity


def get_shape_features(shape_frame, contour, x, y):
    """
    Get the position and orientation of the shape
    Ref: https://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter2.pdf
    """
    moment = cv2.moments(contour)
    centroid, theta, circularity = get_orientation_features(moment)

    # Fit an ellipse to the contour
    shape = cv2.fitEllipse(contour) #TODO: Could also use polygon

    position = shape[0]

    # Get area of the object
    size = moment["m00"]

    length = 100 #TODO: Adjust length
    # Minor Axis
    dx_minor = int(length * np.cos(theta + np.radians(90)))
    dy_minor = int(length * np.sin(theta + np.radians(90)))
    cv2.line(shape_frame, (centroid[0] - dx_minor, centroid[1] - dy_minor), (centroid[0] + dx_minor, centroid[1] + dy_minor), 
             (0, 255, 0), 2)
    cv2.ellipse(shape_frame, shape, (0, 0, 255), 2)
    cv2.putText(shape_frame, f"Area: {size}, Orientation: {degree_to_plaintext(np.degrees(theta))}, Circularity: {circularity}", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return shape_frame, centroid, position, theta, size, circularity


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

def adjust_template_orientation(template, object_orientation, flip_flag=False):
    # Get the center of the template
    center = (template.shape[1] // 2, template.shape[0] // 2)
    _, template_orientation, _ = get_orientation_features(cv2.moments(template))
    # Flip the template if needed
    if flip_flag:
        template = cv2.flip(template, 1)  # Horizontal flip

    # Calculate the angle difference between the template orientation and object orientation
    angle_difference = object_orientation -  np.rad2deg(template_orientation)
    print(f"Angle Difference: {angle_difference}")
    # Rotate the template by the angle difference
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_difference, 1.0)
    adjusted_template = cv2.warpAffine(template, rotation_matrix, (template.shape[1], template.shape[0]))

    return adjusted_template

def generate_template(image_files):
    """ 
    Generate average template from a list of images
    """
    # read the images
    images = [cv2.imread(image) for image in image_files]
    image_contours = []
    for image in images:
        _, _, largest_contour, _, _ = find_skin_color_blobs(image)
        x, y, w, h = cv2.boundingRect(largest_contour)
        largest_contour_roi = image[y:y+h, x:x+w]
        # fill the entire region inside the contour
        filled_roi = np.zeros_like(largest_contour_roi)
        cv2.drawContours(filled_roi, [largest_contour], 0, (255, 255, 255), -1)
        largest_contour_roi = cv2.bitwise_and(largest_contour_roi, filled_roi)
        largest_contour_roi = preprocess(largest_contour_roi)
        image_contours.append(largest_contour_roi)

    target_size = (100, 100)
    largest_contour_rois = [cv2.resize(contour, target_size) for contour in image_contours]
    # binary thresholding
    largest_contour_rois = [cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY) for contour in largest_contour_rois]
    filled_roi = np.zeros_like(largest_contour_rois[0])

    # accumulate the filled rois
    for roi in largest_contour_rois:
        filled_roi += roi

    # normalize the accumulated roi
    filled_roi = (filled_roi / len(largest_contour_rois)).astype(np.uint8)

    largest_contour_rois = [cv2.adaptiveThreshold(contour, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2) for contour in largest_contour_rois]
    largest_contour_rois = [cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR) for contour in largest_contour_rois]
    average_contour = np.mean(largest_contour_rois, axis=0).astype(np.uint8)
    return average_contour

def generate_template_from_mask(image_file, target_size=(200, 200)):
    masks = []
    for im in image_file:
        image = cv2.imread(im)
        _, mask, _, _, _ = find_skin_color_blobs(image)
        mask_resized = cv2.resize(mask, target_size)
        mask_resized = preprocess(mask_resized) # TODO: order of preprocessing impacts the masking output
        masks.append(mask_resized)

    # Create a blank image to accumulate filled mask
    filled_mask = np.zeros(target_size, dtype=np.uint8)

    # Accumulate masks
    for mask in masks:
        filled_mask += mask

    # Normalize accumulated mask
    filled_mask = (filled_mask / len(masks)).astype(np.uint8)

    # Threshold the filled mask
    thresholded_mask = cv2.threshold(filled_mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Convert to BGR for consistency
    average_mask = cv2.cvtColor(thresholded_mask, cv2.COLOR_GRAY2BGR)

    return average_mask


def get_templates(template_directory="../../data/archive/Template/"):
    """
    Function to generate a pyramid of templates
    """
    generated_templates = {}
    # generate different template for each subdirectory
    for i in ["0", "5", "8", "2"]:
        digit = glob.glob(os.path.join(template_directory, f"{i}/*.jpg"))
        temp = generate_template_from_mask([digit[0]]) # TODO 
        generated_templates[i] = temp

    # plot the generated templates
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    for i, (k, v) in enumerate(generated_templates.items()):
        ax[i].imshow(v)
        ax[i].axis("off")
        ax[i].set_title(f"Digit {k}")

    return generated_templates


def custom_template_matching(skin_mask,template_pyramids,  threshold=0.6):
    """
    Custom template matching function that uses the circularity of the digit to determine the best match
    """
    target_pyramids = get_pyramid(skin_mask)
    best_gesture = None
    best_match = 0
    gesture_sum = {k: 0 for k in template_pyramids.keys()}

    for i, pyramid in enumerate(target_pyramids[::-1]):
        template_idx = len(target_pyramids) - i - 1
        img = pyramid.copy()
        # print(f"Image level: {i+1}")
        # print(f"Template idx: {template_idx}")
        for keys in ["0", "5", "8", "2"]:
            curr_template = template_pyramids[keys][template_idx]
            curr_template = cv2.cvtColor(curr_template, cv2.COLOR_BGR2GRAY)
            match = cv2.matchTemplate(img, curr_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match)
            gesture_sum[keys] += max_val
        
        
    gesture_average = {k: v/len(target_pyramids) for k, v in gesture_sum.items()}
    # check if the best average is greater than the threshold
    best_gesture = max(gesture_average, key=gesture_average.get)
    if gesture_average[best_gesture] <= threshold:
        best_gesture = ""
    return gesture_average, best_gesture