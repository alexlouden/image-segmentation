import hashlib


def make_hash(params):
    sha = hashlib.sha1(str(frozenset(params.items())))
    return sha.hexdigest()
