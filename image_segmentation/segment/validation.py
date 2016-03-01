from wtforms import validators
from flask_inputs import Inputs
from flask import Blueprint, jsonify

validation = Blueprint('validation', __name__)


class ValidationError(Exception):
    status_code = 400

    def __init__(self, errors, status_code=None):
        Exception.__init__(self)
        self.errors = errors

        if status_code is not None:
            self.status_code = status_code


@validation.errorhandler(ValidationError)
def handle_invalid_usage(error):
    response = jsonify(success=False, errors=error.errors)
    response.status_code = error.status_code
    return response


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
        # Don't validate number if no cluster_method given
        if not form.data['cluster_method'] and not field.data:
            raise validators.StopValidation()

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
