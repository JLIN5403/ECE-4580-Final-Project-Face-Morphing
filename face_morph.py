#!/usr/bin/env python

import numpy as np
import cv2
import dlib
import random
#import sys


def draw_point(img, p, color):
    cv2.circle(img, p, 4, color, -1)


def detectFaceLandmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    height = image.shape[0]
    width = image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    points = []
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((int(x), int(y)))
    points.append((0, int(0.5 * float(height))))
    points.append((0, 0))
    points.append((int(0.5 * float(width)), 0))
    points.append((width - 1, 0))
    points.append((width - 1, int(0.5 * float(height))))
    points.append((width - 1, height - 1))
    points.append((int(0.5 * float(width)), height - 1))
    points.append((0, height - 1))
    return points


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def delaunayTriangulation(img, subdiv, points, delaunay_color, doDraw):
    triangleList = subdiv.getTriangleList()
    r = (0, 0, img.shape[1], img.shape[0])
    indexesList = []
    for t in triangleList:

        pt1 = (round(t[0]), round(t[1]))
        pt2 = (round(t[2]), round(t[3]))
        pt3 = (round(t[4]), round(t[5]))
        id1 = points.index(pt1)
        id2 = points.index(pt2)
        id3 = points.index(pt3)

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            if doDraw:
                cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
            indexesList.append((id1, id2, id3))
    return indexesList


def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (round(centers[i][0]), round(centers[i][1])), 1, (0, 0, 0), cv2.LINE_AA, 0)


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


if __name__ == '__main__':

    filename1 = 'Chef'
    filename2 = 'CP3'
    alphaList = []
    for n in range(51):
        print(n)
        alphaList.append(float(0.02 * n))
    #alphaList.append(1)

    # Read images
    img1 = cv2.imread(filename1 + '.jpg')
    height1 = float(img1.shape[0])
    width1 = float(img1.shape[1])
    img1 = cv2.resize(img1, (int(0.5 * width1), int(0.5 * height1)))
    img1_orig = img1.copy()
    img2 = cv2.imread(filename2 + '.jpg')
    height2 = float(img2.shape[0])
    width2 = float(img2.shape[1])
    img2 = cv2.resize(img2, (int(img1.shape[1]), int(img1.shape[0])))
    img2_orig = img2.copy()

    points1 = detectFaceLandmarks(img1)
    points2 = detectFaceLandmarks(img2)

    img1_points = img1_orig.copy()
    for p1 in points1:
        draw_point(img1_points, p1, (255, 0, 0))
    img2_points = img2_orig.copy()
    for p2 in points2:
        draw_point(img2_points, p2, (255, 0, 0))

    # Convert Mat to float data type
    #img1 = np.float32(img1)
    #img2 = np.float32(img2)

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D((0, 0, img2.shape[1], img2.shape[0]))

    for p2 in points2:
        subdiv.insert(p2)

    indexesList = delaunayTriangulation(img2, subdiv, points2, (0, 255, 0), False)

    morphVideo = []
    for alpha in alphaList:
        points = []

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = round((1 - alpha) * float(points1[i][0]) + alpha * float(points2[i][0]))
            y = round((1 - alpha) * float(points1[i][1]) + alpha * float(points2[i][1]))
            points.append((x, y))

        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

        for indexes in indexesList:

            x = int(indexes[0])
            y = int(indexes[1])
            z = int(indexes[2])

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

        # Display Result
        morphVideo.append(np.uint8(imgMorph))
    cv2.imshow('Image 1 with Facial Landmarks', img1_points)
    cv2.imshow('Image 2 with Facial Landmarks', img2_points)
    subdiv1 = cv2.Subdiv2D((0, 0, img1.shape[1], img1.shape[0]))

    for p1 in points1:
        subdiv1.insert(p1)
        img1_triangles = img1_orig.copy()
        # Draw delaunay triangles
        indexesList = delaunayTriangulation(img1_triangles, subdiv1, points1, (0, 255, 0), True)
        cv2.imshow('Image 1 with Delaunay Triangulation', img1_triangles)
        cv2.waitKey(100)
    subdiv2 = cv2.Subdiv2D((0, 0, img2.shape[1], img2.shape[0]))

    for p2 in points2:
        subdiv2.insert(p2)
        img2_triangles = img2_orig.copy()
        # Draw delaunay triangles
        indexesList = delaunayTriangulation(img2_triangles, subdiv2, points2, (0, 255, 0), True)
        cv2.imshow('Image 2 with Delaunay Triangulation', img2_triangles)
        cv2.waitKey(100)

    # Allocate space for voronoi Diagram
    img1_voronoi = np.zeros(img1.shape, dtype=img1.dtype)

    # Draw voronoi diagram
    draw_voronoi(img1_voronoi, subdiv1)

    cv2.imshow('Image 1 Voronoi', img1_voronoi)

    # Allocate space for voronoi Diagram
    img2_voronoi = np.zeros(img2.shape, dtype=img2.dtype)

    # Draw voronoi diagram
    draw_voronoi(img2_voronoi, subdiv2)

    cv2.imshow('Image 2 Voronoi', img2_voronoi)

    for morphImage in morphVideo:
        cv2.imshow('Morphed Video', morphImage)
        cv2.waitKey(50)

    cv2.imshow('Midway Face', morphVideo[alphaList.index(0.5)])
    cv2.waitKey(0)
