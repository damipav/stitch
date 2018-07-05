import numpy as np
import os
import cv2

print(cv2.__version__)


class Match:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
        search_params = dict(checks=5)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.warp_size = None

    def _get_SURF_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

    def match(self, path_1, path_2):

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        print(str(path_1) + ":" + str(path_2))

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        print(str(h1) + "," + str(w1))
        print(str(h2) + "," + str(w2))

        feature1 = self._get_SURF_features(img1)
        feature2 = self._get_SURF_features(img2)

        matches = self.flann.knnMatch(feature1['des'], feature2['des'], k=2)
        kps1 = feature1['kp']
        kps2 = feature2['kp']

        goods = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.6 * n.distance and np.fabs(kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0]) < 0.1 * w1:
                goods.append((m.trainIdx, m.queryIdx))

        print(len(goods))

        if len(goods) > 10:
            matched_kps1 = np.float32([kps1[i].pt for (_, i) in goods])
            matched_kps2 = np.float32([kps2[i].pt for (i, _) in goods])

            H, _ = cv2.findHomography(matched_kps1, matched_kps2, cv2.RANSAC, 5.0)

            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            new_corners2 = cv2.perspectiveTransform(corners2, H)

            pts = np.concatenate((corners1, new_corners2), axis=0)
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

            t = [-xmin, -ymin]
            newH = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            im = cv2.warpPerspective(img1, newH.dot(H), (xmax - xmin, ymax - ymin))

            ov_h = (h1 - t[1]) // 2
            im[h1 - ov_h: h1 + h2 - ov_h * 2 , t[0]:w2 + t[0]] = img2[ov_h:, :]

            return im[:-(h1-h2), :]
        else:
            return None


if __name__ == '__main__':
    folder = "data"

    path1 = "data/ortho_rectified_2.jpg"
    path2 = "data/rect_3.jpg"

    res_img = Match().match(path1, path2)
    result_path = 'result.jpg'
    cv2.imwrite(result_path, res_img)
    print("success!")
