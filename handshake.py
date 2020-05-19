import cv2
from google.colab.patches import cv2_imshow
#import winsound
from detectors import *


def con_detect():
    cap = cv2.VideoCapture('boo2.mp4')
    # https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
    cap.set(3, 1280/2)
    cap.set(4, 1024/2)

    HandsDetector = TSDetector()
    #FaceDetector = CVLibDetector()

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = HandsDetector.detect(rgb)

        img_detected = add_objects_to_image(rgb, hands)

        #cv2.imshow('frame', cv2.cvtColor(img_detected, cv2.COLOR_RGB2BGR))
        cv2_imshow(cv2.cvtColor(img_detected, cv2.COLOR_RGB2BGR))
        for hand in hands: 
         if objects_touch(hand,hands):
            try:
              # winsound.PlaySound("audiocheck.net_sin_1000Hz_-3dBFS_0.1s.wav", winsound.SND_NOWAIT)
              print("violation")
            except:
                pass
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    con_detect()
