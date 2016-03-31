from os import environ
from redis import StrictRedis
from redis.exceptions import ConnectionError, TimeoutError
from flask import Flask
from flask.ext.redis import FlaskRedis
from mockredis import MockRedis
from opbeat.contrib.flask import Opbeat

from segment.views import views


class OptionalRedis(StrictRedis):
    """ Redis is optional """

    def safe_get(self, name):
        try:
            return self.get(name)
        except (ConnectionError, TimeoutError) as e:
            print 'Warning: Redis connection error during get: {}'.format(e)
            return None

    def safe_set(self, name, value, **kwargs):
        try:
            return self.set(name, value, **kwargs)
        except (ConnectionError, TimeoutError) as e:
            print 'Warning: Redis connection error during set: {}'.format(e)
            return None


class MockRedisWrapper(MockRedis, OptionalRedis):
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
        redis = FlaskRedis.from_custom_provider(OptionalRedis)

        app.opbeat = Opbeat(
            app,
            organization_id='f4005e762f244a3c9ad94d9f4d4cabad',
            app_id='5d7d348b72',
            secret_token='ed50d6109c233614d041ea29bc48ca2aa1a564c5',
        )

    redis.init_app(app)

    app.register_blueprint(views)
    views.redis = redis

    return app, redis

# For gunicorn
app, _ = create_app()
