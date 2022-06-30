from tensorflow.keras.models import load_model
import numpy as np
import cv2


model = load_model("char_reader/model.hdf5")


def convert_to_char(class_number):
    character = ''
    if class_number == '10':
        character = 'A'
    elif class_number == '11':
        character = 'B'
    elif class_number == '12':
        character = 'C'
    elif class_number == '13':
        character = 'D'
    elif class_number == '14':
        character = 'E'
    elif class_number == '15':
        character = 'F'
    elif class_number == '16':
        character = 'G'
    elif class_number == '17':
        character = 'H'
    elif class_number == '18':
        character = 'J'
    elif class_number == '19':
        character = 'K'
    elif class_number == '20':
        character = 'L'
    elif class_number == '21':
        character = 'M'
    elif class_number == '22':
        character = 'N'
    elif class_number == '23':
        character = 'P'
    elif class_number == '24':
        character = 'Q'
    elif class_number == '25':
        character = 'R'
    elif class_number == '26':
        character = 'S'
    elif class_number == '27':
        character = 'T'
    elif class_number == '28':
        character = 'U'
    elif class_number == '29':
        character = 'V'
    elif class_number == '30':
        character = 'W'
    elif class_number == '31':
        character = 'X'
    elif class_number == '32':
        character = 'Y'
    elif class_number == '33':
        character = 'Z'
    else:
        character = class_number
    return character


def predict_number(number_image):
    # number_image = np.asarray(number_image)
    result = model.predict(number_image)
    result = str(result)
    return convert_to_char(result)


def full_image_reading(list_localized_image):
    result = []
    if len(list_localized_image) == 0:
        return 0
    for i in range(len(list_localized_image)):
        line = []
        for j in range(len(list_localized_image[i])):
            image_resized = cv2.cvtColor(list_localized_image[i][j], cv2.COLOR_BGR2GRAY)
            # print(image_resized.shape)
            # cv2.imshow("image", image_resized)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            image_resized = cv2.resize(image_resized, (20, 40))
            image_resized = np.reshape(image_resized, (20, 40, 1))
            image_resized = image_resized / 255
            image_resized = np.asarray(image_resized)
            img_list = [image_resized]
            img_list = np.asarray(img_list)
            res = model.predict(img_list)
            res = res.argmax(axis=1)
            res = str(res[0])
            res = convert_to_char(res)
            line.append(res)
            # char = predict_number(image_resized)
            # print(char)

        result.append(line)
    return result

