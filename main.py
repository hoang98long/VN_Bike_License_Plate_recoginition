from char_localization import char_localization
from char_reader import char_reader
from plate_detection import object_detection
import cv2
import os

image_dir = "test"


def process():
    files = os.listdir(image_dir)
    for file in files:
        img = cv2.imread(image_dir+'/'+file)
        img_crop = object_detection.cropping(img)
        image_array = char_localization.localizing(img_crop)
        result = char_reader.full_image_reading(image_array)
        print("result: ")
        print(result)
        cv2.imshow("cropped", img_crop)
        cv2.waitKey()
        cv2.destroyAllWindows()



process()




