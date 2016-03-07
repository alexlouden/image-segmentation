from wtforms import validators
from flask_inputs import Inputs
from alg import COLOUR_SPACES, CLUSTER_METHODS


class ValidationError(Exception):
    status_code = 400

    def __init__(self, errors, status_code=None):
        Exception.__init__(self)
        self.errors = errors

        if status_code is not None:
            self.status_code = status_code


def validate_num_clusters_automatic(form, field):

    if form.data['cluster_method'] == 'meanshift':
        if field.raw_data:
            msg = 'num_clusters is automatic with meanshift'
            raise validators.ValidationError(msg)
        else:
            # Don't validate as number
            raise validators.StopValidation()


def validate_num_clusters_required_with_kmeans(form, field):

    clustering_methods = ('kmeans', 'ward')
    if form.data['cluster_method'] in clustering_methods:
        if not field.raw_data:
            msg = 'num_clusters is required with any of {}'.format(', '.join(clustering_methods))
            raise validators.ValidationError(msg)


class StringIntegerRange(validators.NumberRange):

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
        super(StringIntegerRange, self).__call__(form, field)


class StringFloatRange(validators.NumberRange):

    def __call__(self, form, field):
        try:
            # Convert type
            field.data = float(field.data)
        except (ValueError, TypeError):
            raise validators.ValidationError(self.message % dict(min=self.min, max=self.max))

        # Now check range
        super(StringFloatRange, self).__call__(form, field)


class ImageInputs(Inputs):
    args = {
        'colour_space': [
            validators.Optional(),
            validators.AnyOf(
                COLOUR_SPACES,
                message='colour_space must be one of %(values)s'
            )
        ],
        'cluster_method': [
            validators.Optional(),
            validators.AnyOf(
                CLUSTER_METHODS,
                message='cluster_method must be one of %(values)s'
            )
        ],
        'num_clusters': [
            validate_num_clusters_required_with_kmeans,
            validate_num_clusters_automatic,
            StringIntegerRange(1, 100, message='num_clusters must be an integer between %(min)s and %(max)s'),
        ],
        'quantile': [
            validators.Optional(),
            StringFloatRange(0, 1, message='quantile must be a float between  %(min)s and %(max)s'),
        ]
    }
    rule = {
        'image_url': [
            validators.DataRequired('Image URL is missing'),
            validators.URL(require_tld=True, message='Image URL is invalid')
        ]
    }
