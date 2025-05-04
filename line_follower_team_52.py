'''
Created by MR4010.10's team 52:
    Rodrigo Garcia - A01190897
    Roberto Ferro - A01374849
    Owen Jauregui - A01638122

Last updated on 05/04/2025
'''

from controller import Display
import numpy as np
import math
import cv2

# Constants for HoughLinesP
rho = 1
theta = np.pi / 180
thresh = 20
min_length = 10
max_gap = 50

# Scale factor for the control
scale_factor = 0.005

# RoI for the street
roi = np.array([[(0,128),(0,98),(50,80),(206,80),(256,98),(256,128)]], dtype=np.int32)

# Range of HSV for yellow
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

def detect_borders(grey_img):
    """Detects borders from a greyscale image by using Canny edge detection.

    Parameters
    ----------
    grey_img : np.ndarray
        Greyscale image where the borders want to be detected from.

    Returns
    -------
    np.ndarray
        Image with the detected borders from the input greyscale image.
    """
    blur = cv2.GaussianBlur(grey_img, (5, 5), 0, 0)
    return cv2.Canny(blur, 50, 140)

def yellow_filter(bgr, image):
    """Applies a yellow filter to an image using a BGR image as the input for the mask.

    Parameters
    ----------
    bgr : np.ndarray
        BGR image used to calculate the mask for the yellow filter.
    image : np.ndarray
        Image used to calculate the mask for the yellow filter.

    Returns
    -------
    np.ndarray
        Resulting image from applying a yellow filter to image.
    """
    # Calculate mask for yellow pixels
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply mask to image
    return cv2.bitwise_and(image, image, mask=mask) 

def get_roi(image, poly_points):
    """Extracts the region of interest for a particular image and poligons.

    Parameters
    ----------
    image : np.ndarray
        Matrix of the image where the roi is going to be extracted from.
    poly_points : np.ndarray
        List of points for the roi's polygon.

    Returns
    -------
    np.ndarray
        Masked image from the input using the desired roi.
    """
    # Fill the RoI with white values in a new image
    roi_mask = np.zeros_like(image)
    cv2.fillPoly(roi_mask, poly_points, 255)

    # Apply previous mask to the image
    return cv2.bitwise_and(image, image, mask=roi_mask)

def get_lines(border_img):
    """Gets lines from an image of borders using HoughLinesP.

    Parameters
    ----------
    border_img : np.ndarray
        Image containing borders.

    Returns
    -------
    np.ndarray
        Array of lines extracted with the format (x1, y1, x2, y2).
    """
    return cv2.HoughLinesP(border_img, rho, theta, thresh, np.array([]), minLineLength=min_length, maxLineGap=max_gap)

def show_lines(image, lines):
    """Draws an array of lines on a specified image.

    Parameters
    ----------
    image : np.ndarray
        Image to which the lines are going to be drawn to.
    lines : np.ndarray
        Array of lines to be drawn into the image.

    Returns
    -------
    np.ndarray
        Image with the resulting lines drawn.
    """
    display_mask_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        # Cycle through all lines
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Draw lines in red
            cv2.line(display_mask_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return display_mask_with_lines

def calculate_steering_from_lines(lines, image_width):
    """Calculates the steering angle required to follow the lines specified.

    Parameters
    ----------
    lines : np.ndarray
        Array of lines to follow.
    image_width : int
        Width of the image used as reference to calculate the lines.

    Returns
    -------
    float
        The steering value required to follow the lines.
    """
    # No line given safe-guard
    if lines is None:
        return 0.0  

    line_positions = np.zeros_like(lines)
    # Get an x position for each line by averaging x1 and x2
    for i, line in enumerate(lines):
        x1, _, x2, _ = line[0]
        midpoint_x = (x1 + x2) / 2
        line_positions[i] = midpoint_x

    # Average x for all calculated lines
    avg_position = np.mean(line_positions)

    # Calculate offset from the center of the image
    offset = avg_position - (image_width / 2)

    # Calculate the steering angle using the offset to the center and a scaling factor.
    steering_angle = offset * scale_factor

    return steering_angle

def display_image(display, image):
    """Displays a desired image into a Webots display.

    Parameters
    ----------
    display : Display
        Display to which the image will be sent to.
    image : np.ndarray
        Image desired to show.

    Returns
    -------
    None
    """
    # Convert from gray or BGR to RGB
    if len(image.shape) == 2:
        image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display image in Webots
    image_ref = display.imageNew(
        image_display.tobytes(),
        Display.RGB,
        width=image_display.shape[1],
        height=image_display.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

def get_direction(grey_img, bgr_img):
    """Chooses the angle to which the car should steer.

    Parameters
    ----------
    grey_img : np.ndarray
        Greyscale image where the borders want to be detected from.

    Returns
    -------
    float
        Angle in degrees to which the car should steer.
    """
    # Pre-process the input image
    border_image = detect_borders(grey_img) 
    roi_image = get_roi(border_image, roi)
    yellow_img = yellow_filter(bgr_img, roi_image)

    # Calculate the lines
    lines = get_lines(yellow_img)

    # Check if lines are valid
    if lines is None:
        return grey_img, 0
    else:
        # Show lines in image and calculate the steering
        processed_img = show_lines(roi_image, lines)
        return processed_img, calculate_steering_from_lines(lines, grey_img.shape[1])
    
if __name__=='__main__':
    print('This is a custom library, it is not meant to be executed by itself')