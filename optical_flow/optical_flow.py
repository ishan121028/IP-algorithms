import cv2 as cv
import argparse
import numpy as np

def get_optical_flow(prev_image, new_image, dense=True):

    if not dense:
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        

        old_gray = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
        frame_gray = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)

        return img, good_new, good_old

    else:
        prvs = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(prev_image)
        hsv[..., 1] = 255
        next = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr, flow
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optical flow')
    parser.add_argument('--dense', type=bool, default=True)
    parser.add_argument('--webcam', type=bool, default=True)

    args = parser.parse_args()

    if args.webcam:
        capture = cv.VideoCapture(0)
        first_image = True
        while True:
            ret_val, frame = capture.read()
            if first_image:
                first_image = False
                prev_image = frame
                continue
            
            op_f = get_optical_flow(prev_image, frame, args.dense)
            cv.imshow("Optical Flow", op_f)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    else:
        raise Exception("Not implemented")

            
