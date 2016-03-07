import hashlib


def make_dict_hash(params):
    sha = hashlib.sha1(str(frozenset(params.items())))
    return 'd:' + sha.hexdigest()


def make_url_hash(url):
    sha = hashlib.sha1(str(url))
    return 'u:' + sha.hexdigest()
