import cv2
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy

PREDICTOR_PATH = "./model/shape_predictor_68_face_landmarks.dat"
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]                            

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    for i in rects:
        print(i)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

# Swap 2->1
# img1 = cv2.imread("./input/Eddie_Van_Halen_(1993).jpg")
img1 = cv2.imread("./input/a1.jpg")
# img1 = cv2.resize(img1, (img1.shape[1]*1, img1.shape[0]*1))

# img2 = cv2.imread("./input/597px-Ed_Miliband.jpg")
img2 = cv2.imread("./input/a2.jpg")
# img2 = cv2.resize(img2, (img2.shape[1]*1, img2.shape[0]*1))

landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2)

M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
print(M)

warped_im2 = warp_im(img2, M, img1.shape)
warped_corrected_im2 = correct_colours(img1, warped_im2, landmarks1)

mask = get_face_mask(img2, landmarks2)
warped_mask = warp_im(mask, M, img1.shape)
combined_mask = numpy.max([get_face_mask(img1, landmarks1), warped_mask],
                          axis=0)

output_im = img1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
plt.axis('off')
# cv2.imshow("", output_im.astype('float64'))
cv2.imwrite('./output/output.jpg', output_im)

# Swap 1 -> 2
# img1 = cv2.imread("./input/597px-Ed_Miliband.jpg")
img1 = cv2.imread("./input/a2.jpg")
# img1 = cv2.resize(img1, (img1.shape[1]*1, img1.shape[0]*1))

# img2 = cv2.imread("./input/Eddie_Van_Halen_(1993).jpg")
img2 = cv2.imread("./input/a1.jpg")
# img2 = cv2.resize(img2, (img2.shape[1]*1, img2.shape[0]*1))

landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2)

M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
print(M)

warped_im2 = warp_im(img2, M, img1.shape)
warped_corrected_im2 = correct_colours(img1, warped_im2, landmarks1)

mask = get_face_mask(img2, landmarks2)
warped_mask = warp_im(mask, M, img1.shape)
combined_mask = numpy.max([get_face_mask(img1, landmarks1), warped_mask],
                          axis=0)

output_im = img1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
plt.axis('off')
# cv2.imshow("", output_im.astype('float64'))
cv2.imwrite('./output/output1.jpg', output_im)