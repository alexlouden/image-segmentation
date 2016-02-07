from __future__ import division
import os
# import matplotlib.pyplot as plt
import numpy as np
import requests
import cv2
from sklearn.cluster import MeanShift, KMeans, DBSCAN, estimate_bandwidth


job = {
    'image_url': '??',
    'colour_space': 'hsv',
    'cluster': 'k-means',
    'num_clusters': 10
}


def convert_image_from_bgr(image, colour_space_to):
    COLOUR_CONVERSIONS = {
        'rgb': cv2.COLOR_BGR2RGB,
        'hsv': cv2.COLOR_BGR2HSV,
        'hls': cv2.COLOR_BGR2HLS,
        'ycrcb': cv2.COLOR_BGR2YCR_CB,
        'lab': cv2.COLOR_BGR2LAB,
        'luv': cv2.COLOR_BGR2LUV,
    }
    conversion = COLOUR_CONVERSIONS.get(colour_space_to)
    return cv2.cvtColor(image, conversion)


def convert_image_to_bgr(image, colour_space_from):
    COLOUR_CONVERSIONS = {
        'rgb': cv2.COLOR_RGB2BGR,
        'hsv': cv2.COLOR_HSV2BGR,
        'hls': cv2.COLOR_HLS2BGR,
        'ycrcb': cv2.COLOR_YCR_CB2BGR,
        'lab': cv2.COLOR_LAB2BGR,
        'luv': cv2.COLOR_LUV2BGR,
    }
    conversion = COLOUR_CONVERSIONS.get(colour_space_from)
    return cv2.cvtColor(image, conversion)


def download_image(url):
    response = requests.get(url, stream=True)
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


class Parameters(object):
    pass


MAX_DIMENSION = 200


class ClusterJob(object):
    def __init__(self, image_url, colour_space, cluster_method, scale=None, n_clusters=None, quantile=None):
        self.url = image_url
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
        if n_clusters is None:
            self.params.n_clusters = 8
        else:
            # TODO validate
            self.params.n_clusters = n_clusters

        # Mean-shift param
        if quantile is None:
            self.params.quantile = 0.1
        else:
            self.params.quantile = quantile

        # DBSCAN param
        # if epsilon is None:
        self.params.epsilon = 2

        self.validate()

    def validate(self):
        pass
        # validate_url(self.url)
        # validate colour space
        # validate cluster method and options

    def process(self):
        self.fetch_image()
        self.scale()
        self.cluster()

    def fetch_image(self):
        self.image = download_image(self.url)

        # self.image_height
        # Colour channels? 4 - alpha, 1 grey, 3 = bgr?
        # Resize if too big?
        # E.g. downscale_local_mean

    def scale(self):

        self.original_image = self.image.copy()

        self.image_height, self.image_width = self.image.shape[:2]
        print self.image_width, self.image_height

        if max(self.image_width, self.image_height) > MAX_DIMENSION:
            # Need to shrink

            if self.image_width > self.image_height:
                new_width = MAX_DIMENSION
                new_height = int(self.image_height * new_width / self.image_width)
            else:
                new_height = MAX_DIMENSION
                new_width = int(self.image_width * new_height / self.image_height)

            self.image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.image_height, self.image_width = self.image.shape[:2]
            print self.image_width, self.image_height

    def cluster(self):

        # Convert from BGR to desired colour space
        self.image = convert_image_from_bgr(self.original_image, self.colour_space)

        # Convert into columns of colours
        image_cols = self.image.reshape(-1, 3).astype(np.float)

        # Scale
        for i in range(0, 3):
            image_cols[:, i] *= self.params.scale[i]

        # Cluster
        if self.cluster_method == 'k-means':
            segmented = self.cluster_k_means(image_cols)

        elif self.cluster_method == 'mean-shift':
            segmented = self.cluster_means_shift(image_cols)

        elif self.cluster_method == 'dbscan':
            segmented = self.cluster_dbscan(image_cols)
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
            n_clusters=self.params.n_clusters,
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
        for i in range(0, np.max(db.labels_) + 1):
            cluster_points = image_cols[db.labels_ == i]
            cluster_mean = np.mean(cluster_points, axis=0)
            centers[i, :] = cluster_mean

        print centers
        centers = self.unscale_centers(centers)

        # TODO assign noise to nearest cluster?

        labels = db.labels_
        segmented = centers[labels]
        return segmented

    def export_segmented_image(self, filename):

        cv2.imwrite(filename, self.segmented_image, (cv2.IMWRITE_JPEG_QUALITY, 80))


if __name__ == "__main__":
    # url ="https://www.google.com.au/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"

    cluster = 'affinity-propogation'
    colour = 'hsv'

    for cluster in ['dbscan']:
        for colour in ['rgb', 'hsv', 'ycrcb', 'hls', 'lab', 'luv']:
            cj = ClusterJob("", colour, cluster, n_clusters=7, quantile=0.05, scale=(1, 1, 1))
            # cj.fetch_image()
            cj.image = load_image("~//Projects/image_segmentation/golden.jpeg")
            # show(cj.image)
            cj.scale()
            cj.cluster()
            # show(cj.segmented_image)
            cj.export_segmented_image('golden_{}_{}_s221.jpg'.format(cluster, colour))
