import cv2
import numpy as np
import os
import time

def save_point_details(x=int, y=int):
    with open('points_log.txt', 'a') as log_file:
        log_file.write(f'Pos: X={x}, Y={y}\n')

def enhance_image():
    source_img = cv2.imread('source_variant.png', cv2.IMREAD_COLOR)
    enhanced_img = cv2.GaussianBlur(source_img, (15, 15), 0)
    cv2.imwrite('enhanced_variant.png', enhanced_img)
    print('Enhanced image created successfully!')

def delete_file():
    target_file = 'obsolete_image.png'
    if os.path.exists(target_file):
        os.remove(target_file)
        print('Obsolete image removed.')
    else:
        print('No obsolete image to remove.')

def capture_and_process():
    print('To exit, press "q"!')
    x_coords, y_coords = [], []
    DETECTION_LIMIT = 0.7
    stream = cv2.VideoCapture(0)
    target_pattern = cv2.imread('target_image.jpg', 0)
    pattern_height, pattern_width = target_pattern.shape

    while True:
        success, frame = stream.read()
        if not success:
            continue

        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = cv2.matchTemplate(grey_frame, target_pattern, cv2.TM_CCOEFF_NORMED)
        positions = np.where(detection >= DETECTION_LIMIT)

        for pos in zip(*positions[::-1]):
            center_x, center_y = pos[0] + pattern_width // 2, pos[1] + pattern_height // 2
            cv2.circle(frame, (center_x, center_y), pattern_height // 2, (255, 0, 0), 2)
            save_point_details(center_x, center_y)
            x_coords.append(center_x)
            y_coords.append(center_y)

        cv2.imshow('Detection View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if x_coords and y_coords:
                avg_x = sum(x_coords) // len(x_coords)
                avg_y = sum(y_coords) // len(y_coords)
                print(f'Average Center: X={avg_x}, Y={avg_y}')
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    enhance_image()
    capture_and_process()
    # delete_file()
