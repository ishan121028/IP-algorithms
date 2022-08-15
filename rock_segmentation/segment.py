import argparse
import cv2 as cv
import numpy as np

def rock_segment(image):
    # https://www.semanticscholar.org/paper/A-FRAMEWORK-FOR-AUTOMATED-ROCK-SEGMENTATION-OF-THE-Song-Shan/5e5a27a077db5d8c6c3d46fc44691f1131893796
    # 
    t_lower = 50  # Lower Threshold
    t_upper = 200  
    aperture_size = 5
    
    edge = cv.Canny(image, t_lower, t_upper, 
                    apertureSize=aperture_size)
    # TODO: contour detection

    return None


def classification(image):
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optical flow')
    parser.add_argument('--webcam', type=bool, default=False)
    parser.add_argument('--sample_image', type=bool, default=True)
    parser.add_argument('--segment', type=bool, default=True)
    parser.add_argument('--classify', type=bool, default=True)
    
    args = parser.parse_args()

    if args.test_image:
        frame = cv.imread('sample.png')
        segmented_image = rock_segment(frame)
        cv.imshow("Segmented Image", segmented_image)
        

    if args.webcam:
        capture = cv.VideoCapture(0)
        while True:
            ret_val, frame = capture.read()
            if args.segment:
                segmented_image = rock_segment(frame)
            cv.imshow("Segmented Image", segmented_image)
            
            if args.classify:
                raise Exception("Not implemented yet")
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break