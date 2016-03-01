import unittest
from flask.ext.testing import TestCase

from segment import create_app as seg_create_app


class MyTest(TestCase):

    def create_app(self):

        app = seg_create_app(testing=True)
        return app

    def test_root(self):
        r = self.client.get('/test')
        print r.data
        assert r.status_code == 400


if __name__ == '__main__':
    unittest.main()
