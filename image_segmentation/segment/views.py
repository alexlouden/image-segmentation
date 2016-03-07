import numpy as np
import pickle
from flask import Blueprint, request, jsonify, send_file
from requests.exceptions import HTTPError, Timeout, RequestException

from alg import ClusterJob, download_image, image_to_file
from utils import make_dict_hash, make_url_hash
from validation import ImageInputs, ValidationError


views = Blueprint('views', __name__)


@views.errorhandler(ValidationError)
def handle_invalid_usage(error):
    response = jsonify(success=False, errors=error.errors)
    response.status_code = error.status_code
    return response


def download_image_validate(image_url):

    # Now download image and check type
    try:
        image = download_image(image_url)

    except HTTPError as e:
        # Non-2xx status code
        errors = ['URL returned a {} status code'.format(e.response.status_code)]
        raise ValidationError(errors)

    except Timeout as e:
        errors = ['URL timed out']
        raise ValidationError(errors)

    except RequestException as e:
        # TODO log other errors
        print 'misc error'
        print image_url, e
        errors = ['URL is not a valid image']
        raise ValidationError(errors)

    except Exception:
        # TOOD catch these
        raise

    if image is None:
        errors = ['URL is not a valid image']
        raise ValidationError(errors)

    return image


def get_image(url):

    # Check image cache
    image_url_hash = make_url_hash(url)
    existing_image = views.redis.get(image_url_hash)
    if existing_image:
        # Load
        image = from_redis(existing_image)
        print 'Cache hit for downloaded image'
    else:
        # Download and validate image
        image = download_image_validate(url)

    return image


def to_redis(image):
    return pickle.dumps(image, protocol=0)


def from_redis(image_data):
    return pickle.loads(image_data)


def return_image(image):
    f = image_to_file(image)
    return send_file(f, mimetype='image/jpeg')


@views.route('/', defaults={'image_url': ''})
@views.route('/<path:image_url>')
def image(image_url=''):

    # Validate
    inputs = ImageInputs(request)
    if not inputs.validate():
        raise ValidationError(inputs.errors)

    params = {
        'url': image_url,
        'args': request.args
    }
    # print params
    params_hash = make_dict_hash(params)

    # Have we already processed this image?
    existing_image = views.redis.get(params_hash)
    if existing_image:
        print 'Cache hit for segmented image'
        image = from_redis(existing_image)
        return return_image(image)

    image = get_image(image_url)
    # print 'dimensions', image.shape

    # drop extra params
    allowed_keys = ('colour_space', 'cluster_method', 'num_clusters', 'quantile')
    kwargs = {k: v for k, v in params['args'].iteritems() if k in allowed_keys}

    cj = ClusterJob(image, **kwargs)
    cj.scale()

    # Cache resized image
    image_url_hash = make_url_hash(image_url)
    views.redis.set(image_url_hash, to_redis(cj.image))

    # TODO time
    cj.cluster()

    # Cache result
    views.redis.set(params_hash, to_redis(cj.segmented_image))

    return return_image(cj.segmented_image)

    # Cache results
    # views.redis.set(params_hash, params)

# TODO catch 20s timeout, complete job on worker?
