import unittest
from flask.ext.testing import TestCase

from segment import create_app as seg_create_app


class SegmentTestCase(TestCase):

    def create_app(self):
        app, redis = seg_create_app(testing=True)
        self.redis = redis
        return app


class TestHappy(SegmentTestCase):

    def test_valid_image(self):
        url = 'https://httpbin.org/image/jpeg'
        r = self.client.get('/' + url)
        self.assert_200(r)
        print self.redis.keys()
        import ipdb; ipdb.set_trace()


class TestValidation(SegmentTestCase):

    def test_root(self):
        r = self.client.get('/')
        self.assertEquals(
            r.json, {
                'success': False,
                'errors': ["Image URL is missing"]
            }
        )
        self.assert_400(r)

    def test_image_500(self):
        url = 'http://httpbin.org/status/500'
        r = self.client.get('/' + url)
        self.assert_400(r)
        self.assertEquals(
            r.json, {
                'success': False,
                'errors': ["URL returned a 500 status code"]
            }
        )

    def test_file_url(self):
        r = self.client.get('/file://blah.jpg')
        self.assertEquals(
            r.json, {
                'success': False,
                'errors': ["URL is not a valid image"]
            }
        )
        self.assert_400(r)

    def test_invalid_url(self):
        r = self.client.get('/invalid_url')
        self.assertEquals(
            r.json, {
                'success': False,
                'errors': ["Image URL is invalid"]
            }
        )
        self.assert_400(r)

# TODO test caching output
# TODO test caching resized image

if __name__ == '__main__':
    unittest.main()
