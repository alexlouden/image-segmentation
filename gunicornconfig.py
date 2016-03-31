# Gunicorn config file

worker_class = 'gevent'
worker_connections = 100
threads = 2
timeout = 27
log_file = ''


def worker_abort(worker):
    print 'aborting', worker
