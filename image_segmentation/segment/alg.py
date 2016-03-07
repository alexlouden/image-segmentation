from __future__ import division
import os
# import matplotlib.pyplot as plt
import numpy as np
import requests
import cv2
from cStringIO import StringIO

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import MeanShift, KMeans, DBSCAN, AgglomerativeClustering, estimate_bandwidth

COLOUR_CONVERSIONS_FROM_BGR = {
    'rgb': cv2.COLOR_BGR2RGB,
    'hsv': cv2.COLOR_BGR2HSV,
    'hls': cv2.COLOR_BGR2HLS,
    'ycrcb': cv2.COLOR_BGR2YCR_CB,
    'lab': cv2.COLOR_BGR2LAB,
    'luv': cv2.COLOR_BGR2LUV,
}
COLOUR_CONVERSIONS_TO_BGR = {
    'rgb': cv2.COLOR_RGB2BGR,
    'hsv': cv2.COLOR_HSV2BGR,
    'hls': cv2.COLOR_HLS2BGR,
    'ycrcb': cv2.COLOR_YCR_CB2BGR,
    'lab': cv2.COLOR_LAB2BGR,
    'luv': cv2.COLOR_LUV2BGR,
}
COLOUR_SPACES = ('rgb', 'hsv', 'hls', 'ycrcb', 'lab', 'luv')
CLUSTER_METHODS = ('kmeans', 'meanshift', 'ward')


def convert_image_from_bgr(image, colour_space_to):
    conversion = COLOUR_CONVERSIONS_FROM_BGR.get(colour_space_to)
    return cv2.cvtColor(image, conversion)


def convert_image_to_bgr(image, colour_space_from):

    conversion = COLOUR_CONVERSIONS_TO_BGR.get(colour_space_from)
    return cv2.cvtColor(image, conversion)


def download_image(url):
    response = requests.get(url, stream=True, timeout=5)
    # TODO use grequests
    # Raise exception on error
    response.raise_for_status()
    numpy_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
    image = cv2.imdecode(numpy_array, cv2.CV_LOAD_IMAGE_COLOR)
    # TODO: handle transparency (load using cv2.IMREAD_UNCHANGED and convert alpha layer to white?)
    return image


def load_image(filename):
    filename = os.path.expanduser(filename)
    filename = os.path.normpath(filename)
    return cv2.imread(filename, cv2.CV_LOAD_IMAGE_COLOR)


def show(image):
    # Show BGR image
    cv2.imshow("Window", image)
    cv2.waitKey()


def image_to_file(image):
    f = StringIO()
    ret, buf = cv2.imencode('.jpg', image)
    f.write(np.array(buf).tostring())
    f.seek(0)
    return f


class Parameters(object):
    pass


MAX_DIMENSION = 500


class ClusterJob(object):
    def __init__(self, image, colour_space='hsv', cluster_method='ward', scale=None, num_clusters=None, quantile=None):
        self.image = image
        self.colour_space = colour_space
        self.cluster_method = cluster_method

        self.params = Parameters()

        # Scaling colour space
        if scale is None:
            self.params.scale = (1, 1, 1)
        else:
            # TODO validate 3 float tuple
            self.params.scale = scale

        # K-means param
        if num_clusters is None:
            self.params.num_clusters = 8
        else:
            # TODO validate
            self.params.num_clusters = int(num_clusters)

        # Mean-shift param
        if quantile is None:
            self.params.quantile = 0.1
        else:
            self.params.quantile = float(quantile)

        # DBSCAN param
        # if epsilon is None:
        self.params.epsilon = 255*0.1

        # Log
        h, w = self.image.shape[:2]
        msg = 'Clustering a {}x{} image: cluster_method={} colour_space={} num_clusters={} quantile={}'.format(
            w, h, cluster_method, colour_space, num_clusters, quantile
        )
        print msg

    def scale(self):

        self.original_image = self.image.copy()

        self.image_height, self.image_width = self.image.shape[:2]

        if max(self.image_width, self.image_height) > MAX_DIMENSION:
            # Need to shrink

            if self.image_width > self.image_height:
                new_width = MAX_DIMENSION
                new_height = int(self.image_height * new_width / self.image_width)
            else:
                new_height = MAX_DIMENSION
                new_width = int(self.image_width * new_height / self.image_height)

            print 'Resizing to {}x{}'.format(new_width, new_height)

            self.image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.image_height, self.image_width = self.image.shape[:2]

    def cluster(self):

        # Convert from BGR to desired colour space
        self.image = convert_image_from_bgr(self.image, self.colour_space)

        # Convert into columns of colours
        image_cols = self.image.reshape(-1, 3).astype(np.float)

        # Scale
        for i in range(0, 3):
            image_cols[:, i] *= self.params.scale[i]

        # Cluster
        if self.cluster_method == 'kmeans':
            segmented = self.cluster_k_means(image_cols)

        elif self.cluster_method == 'meanshift':
            segmented = self.cluster_means_shift(image_cols)

        # Too slow
        # elif self.cluster_method == 'dbscan':
        #     segmented = self.cluster_dbscan(image_cols)

        elif self.cluster_method == 'ward':
            segmented = self.cluster_ward(image_cols)
        else:
            raise RuntimeError('Invalid clustering algorithm')

        # Segmented image in clustered colour space
        segmented_image = segmented.reshape(self.image.shape).astype(np.uint8)

        # Convert back to BGR
        self.segmented_image = convert_image_to_bgr(segmented_image, self.colour_space)

    def unscale_centers(self, centers):

        for i in range(0, 3):
            centers[:, i] /= self.params.scale[i]

        return centers

    def cluster_k_means(self, image_cols):
        print 'K-means clustering'

        km = KMeans(
            n_clusters=self.params.num_clusters,
            max_iter=300
        )
        km.fit(image_cols)

        self.number_of_clusters = km.n_clusters
        print 'number of clusters', self.number_of_clusters

        centers = self.unscale_centers(km.cluster_centers_)

        labels = km.predict(image_cols)
        segmented = centers[labels]

        return segmented

    def cluster_means_shift(self, image_cols):
        print 'Means shifting'

        bandwidth = estimate_bandwidth(image_cols, quantile=self.params.quantile, n_samples=400)
        print self.params.quantile, bandwidth
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=50)
        ms.fit(image_cols)

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        self.number_of_clusters = len(np.unique(ms.labels_))

        print 'number of clusters', self.number_of_clusters

        centers = self.unscale_centers(ms.cluster_centers_)

        labels = ms.predict(image_cols)
        segmented = centers[labels]
        return segmented

    def cluster_dbscan(self, image_cols):
        print 'DBSCAN'
        # TODO handle outliers/noise
        # Look at different metrics?

        db = DBSCAN(eps=self.params.epsilon, min_samples=10, metric='euclidean')
        db.fit(image_cols)

        # from IPython import embed; embed(); import ipdb; ipdb.set_trace()
        self.number_of_clusters = np.max(db.labels_) + 1
        # Ignore -1 cluster, it's noise

        print 'number of clusters', self.number_of_clusters

        # Clusters
        centers = np.zeros((self.number_of_clusters, 3))
        for i in range(0, self.number_of_clusters):
            cluster_points = image_cols[db.labels_ == i]
            cluster_mean = np.mean(cluster_points, axis=0)
            centers[i, :] = cluster_mean

        print centers
        centers = self.unscale_centers(centers)

        # TODO assign noise to nearest cluster?

        labels = db.labels_
        segmented = centers[labels]
        return segmented

    def cluster_ward(self, image_cols):

        # Connectivity
        # TODO optional connectivity
        connectivity = grid_to_graph(*self.image.shape[:2])

        ward = AgglomerativeClustering(
            n_clusters=self.params.num_clusters,
            linkage='ward',
            connectivity=connectivity
        )
        ward.fit(image_cols)

        self.number_of_clusters = len(np.unique(ward.labels_))
        print 'number of clusters', self.number_of_clusters

        centers = np.zeros((self.number_of_clusters, 3))
        for i in range(0, self.number_of_clusters):
            cluster_points = image_cols[ward.labels_ == i]
            cluster_mean = np.mean(cluster_points, axis=0)
            centers[i, :] = cluster_mean

        centers = self.unscale_centers(centers)

        labels = ward.labels_
        segmented = centers[labels]
        return segmented

    def export_segmented_image(self, filename):

        cv2.imwrite(filename, self.segmented_image, (cv2.IMWRITE_JPEG_QUALITY, 80))


if __name__ == "__main__":
    # url ="https://www.google.com.au/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"

    for cluster in ['ward']:
        for colour in ['rgb', 'hsv', 'ycrcb', 'hls']:  # 'hsv', 'ycrcb', 'hls', 'lab', 'luv'
            scale = (1, 1, 1)
            cj = ClusterJob("", colour, cluster, num_clusters=7, quantile=0.05, scale=scale)
            # cj.fetch_image()
            example = 'golden'
            cj.image = load_image("~/Projects/computer-vision/image_segmentation/examples/{}.jpg".format(example))
            # show(cj.image)
            cj.scale()
            cj.cluster()
            # show(cj.segmented_image)
            cj.export_segmented_image('{}_{}_{}_{}{}{}.jpg'.format(example, cluster, colour, *scale))
