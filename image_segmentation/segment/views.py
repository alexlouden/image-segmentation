from flask import Blueprint, request, jsonify, send_file
from requests.exceptions import HTTPError, Timeout, RequestException

from alg import ClusterJob, download_image
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


@views.route('/', defaults={'image_url': ''})
@views.route('/<path:image_url>')
def image(image_url=''):

    params = {
        'url': image_url,
    }
    # print params
    # print request.args
    # print request.view_args

    params_hash = make_dict_hash(params)
    # print params_hash

    # Have we already processed this image?
    existing_image_data = views.redis.get(params_hash)
    if existing_image_data:
        return existing_image_data

    # Validate
    inputs = ImageInputs(request)
    if not inputs.validate():
        response = jsonify(success=False, errors=inputs.errors)
        response.status_code = 400
        return response

    # TODO check image cache

    # Download and validate image
    image = download_image_validate(image_url)

    # print 'dimensions', image.shape

    # Set defaults to missing params
    params['args'] = request.args

    kwargs = {}

    if 'colour_space' in params['args']:
        kwargs['colour_space'] = params['args']['colour_space']

    if 'cluster_method' in params['args']:
        kwargs['cluster_method'] = params['args']['cluster_method']

    if 'num_clusters' in params['args']:
        kwargs['num_clusters'] = int(params['args']['num_clusters'])

    # n_clusters=7,
    # quantile=0.05

    # ward
    # meanshift
    # kmeans

    cj = ClusterJob(image, **kwargs)
    cj.scale()

    # Cache resized image
    # TODO cj.image

    cj.cluster()
    f = cj.export_segmented_image_file()
    return send_file(f, mimetype='image/jpeg')


    # return jsonify(success=True, params=params)
    # request.args.get('', '')

    # Process

    # Cache results
    # app.redis.set(params_hash, params)

# TODO catch 20s timeout, complete job on worker?
