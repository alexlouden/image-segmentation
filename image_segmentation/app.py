import hashlib
from wtforms import validators
from os import environ
from flask import Flask, request, jsonify
from flask.ext.redis import FlaskRedis
from flask_inputs import Inputs

from alg import download_image


app = Flask(__name__)
redis_url = environ.get('REDIS_URL')
if redis_url:
    app.config.setdefault('REDIS_URL', redis_url)

redis = FlaskRedis(app)


def make_hash(params):
    sha = hashlib.sha1(str(frozenset(params.items())))
    return sha.hexdigest()


def format_image(image_data):
    return image_data


def validate_num_clusters_automatic(form, field):

    automatic_clustering_methods = ('ward', 'meanshift')
    if form.data['cluster_method'] in automatic_clustering_methods:
        if field.raw_data:
            msg = 'num_clusters is automatic with any of {}'.format(', '.join(automatic_clustering_methods))
            raise validators.ValidationError(msg)
        else:
            # Don't validate as number
            raise validators.StopValidation()


def validate_num_clusters_required_with_kmeans(form, field):

    if form.data['cluster_method'] == 'kmeans':
        if not field.raw_data:
            msg = 'num_clusters is required with kmeans'
            raise validators.ValidationError(msg)


class StringNumberRange(validators.NumberRange):

    def __call__(self, form, field):
        try:
            # Convert type
            field.data = int(field.data)
        except (ValueError, TypeError):
            # self.message must be defined
            raise validators.ValidationError(self.message % dict(min=self.min, max=self.max))

        # Now check range
        super(StringNumberRange, self).__call__(form, field)


class ImageInputs(Inputs):
    args = {
        'colour_space': [
            validators.Optional(),
            validators.AnyOf(
                ['rgb', 'hsv'],
                message='colour_space must be one of %(values)s'
            )
        ],
        'cluster_method': [
            validators.Optional(),
            validators.AnyOf(
                ['kmeans', 'meanshift', 'ward'],
                message='cluster_method must be one of %(values)s'
            )
        ],
        'num_clusters': [
            validate_num_clusters_required_with_kmeans,
            validate_num_clusters_automatic,
            StringNumberRange(1, 20, message='num_clusters must be an integer between %(min)s and %(max)s'),
        ]

    }
    rule = {
        'image_url': [
            validators.DataRequired('Image URL is missing'),
            validators.URL(require_tld=True, message='Image URL is invalid')
        ]
    }


@app.route('/')
@app.route('/<path:image_url>')
def image(image_url=''):

    # TODO strip args from image_url

    params = {
        'url': image_url,
    }
    # print params
    print request.args
    # print request.view_args

    params_hash = make_hash(params)
    # print params_hash

    # Have we already processed this image?
    # existing_image_data = redis.get(params_hash)
    # if existing_image_data:
    #     return format_image(existing_image_data)

    # Type conversion for numbers
    # try:
    #     args = dict(request.args)
    #     args['num_clusters'] = int(request.args['num_clusters'])
    #     request.args = args
    # except (ValueError, KeyError):
    #     pass

    # Validate
    inputs = ImageInputs(request)
    if not inputs.validate():
        response = jsonify(success=False, errors=inputs.errors)
        response.status_code = 400
        return response

    # Now download image and check type
    try:
        image = download_image(image_url)
    except Exception:
        # TODO catch and throw validation error
        raise
        errors = ['URL is not a valid image']
        response = jsonify(success=False, errors=errors)
        response.status_code = 400
        return response

    if image is None:
        errors = ['URL is not a valid image']
        response = jsonify(success=False, errors=errors)
        response.status_code = 400
        return response

    print 'dimensions', image.shape

    # Set defaults to missing params
    params['args'] = request.args

    return jsonify(success=True, params=params)
    # request.args.get('', '')

    # Process

    # Save
    redis.set(params_hash, params)

# TODO catch 20s timeout, complete job on worker?

if __name__ == "__main__":
    app.run(debug=True)
