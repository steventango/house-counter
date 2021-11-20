import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Change these constants to adjust the selection heuristic.
MIN_HOUSE_AREA = 20
MAX_HOUSE_AREA = 1200
HOUSE_W_H_RATIO = 10

# HSV ranges used for the mask
LOWER_MASK = np.array([0, 0, 0])
UPPER_MASK = np.array([255, 10, 255])

def select(contour):
    """Selection heuristic function for contours.
    Requires a contour to have a m00 moment > 0,
    area within MIN_HOUSE_AREA and MAX_HOUSE_AREA,
    and a width/height ratio < HOUSE_W_H_RATIO.

    Args:
        contour: The contour to test.

    Returns:
        True if the contour is a house, False otherwise.
    """
    M = cv.moments(contour)
    A = cv.contourArea(contour)
    x, y, w, h = cv.boundingRect(contour)
    ratio = w / h if w > h else h / w
    return M['m00'] > 0 and MIN_HOUSE_AREA < A and A < MAX_HOUSE_AREA and ratio < HOUSE_W_H_RATIO


def main():
    img = cv.imread('map.png')

    # Apply blur to reduce noise
    img = cv.GaussianBlur(img, (5, 5), 0)

    # Convert to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Use HSV to mask out the background
    imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(imghsv, LOWER_MASK, UPPER_MASK)

    contours, hierarchy = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    filtered_countours = filter(select, contours)
    house_contours = list(filtered_countours)

    # Randomly generate a color for each house
    colors = list(np.random.random(size=(len(house_contours), 3)) * 256)

    for (i, c) in tqdm(enumerate(house_contours), total=len(house_contours)):
        M = cv.moments(c)
        digit = str(i + 1)
        cx = int(M['m10'] / M['m00']) - 2 * len(digit)
        cy = int(M['m01'] / M['m00'])
        cv.putText(
            img,
            text=digit,
            org=(cx, cy),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.25,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv.LINE_AA
        )

        cv.drawContours(img, house_contours, i, colors[i], 1)

    plt.imshow(img)

    plt.title('Map')
    plt.suptitle(f'House Count: {len(house_contours)}')
    plt.savefig('out.png', dpi=1200)


if __name__ == '__main__':
    main()