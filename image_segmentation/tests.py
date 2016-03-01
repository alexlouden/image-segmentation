import unittest
from flask.ext.testing import TestCase

from segment import create_app as seg_create_app


class MyTest(TestCase):

    def create_app(self):

        app = seg_create_app(testing=True)
        return app

    def test_root(self):
        r = self.client.get('/')
        self.assertEquals(
            r.json, {
                'success': False,
                'errors': ["Image URL is missing"]
            }
        )
        self.assert_400(r)

    def test_valid_image(self):
        url = 'https://httpbin.org/image/jpeg'
        r = self.client.get('/' + url)
        self.assert_200(r)

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


if __name__ == '__main__':
    unittest.main()
