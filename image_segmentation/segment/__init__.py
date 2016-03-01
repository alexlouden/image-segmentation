from os import environ
from flask import Flask
from flask.ext.redis import FlaskRedis

app = Flask(__name__)

redis_url = environ.get('REDIS_URL')
if redis_url:
    app.config.setdefault('REDIS_URL', redis_url)

redis = FlaskRedis(app)


import segment.views
# import app.validation
