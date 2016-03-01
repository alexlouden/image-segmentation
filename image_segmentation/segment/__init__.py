from os import environ
from flask import Flask
from flask.ext.redis import FlaskRedis
from mockredis import MockRedis

from segment.views import views
from segment.validation import validation


class MockRedisWrapper(MockRedis):
    '''A wrapper to add the `from_url` classmethod'''
    @classmethod
    def from_url(cls, *args, **kwargs):
        return cls()


def create_app(testing=False):
    app = Flask(__name__)

    if testing:
        app.config['TESTING'] = True

    # Redis config
    redis_url = environ.get('REDIS_URL')
    if redis_url:
        app.config.setdefault('REDIS_URL', redis_url)

    # Mock redis
    if app.testing:
        redis = FlaskRedis.from_custom_provider(MockRedisWrapper)
    else:
        redis = FlaskRedis()

    redis.init_app(app)

    app.register_blueprint(views)
    app.register_blueprint(validation)

    return app
