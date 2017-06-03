# Image segmentation

Image segmentation web service designed for hosting on Heroku.

![](http://i.imgur.com/hsiR4hV.jpg)
![](http://segment-image.herokuapp.com/http://i.imgur.com/hsiR4hV.jpg?cluster_method=meanshift&quantile=0.015&colour_space=ycrcb)

The API is inspired by [imgix](http://imgix.com) - you pass in the image url (optionally urlencoded),  control the algorithm with query parameters, and recieve a segmented image as the response (or a JSON formatted error). This allows you to embed it in a web page, or play with the algorithm in your browser without having to install any software.

```bash
https://segment-image.herokuapp.com/<image_url>?cluster_method=kmeans&num_clusters=10
```

The service supports the following parameters:

**cluster_method** ward, meanshift or kmeans<br>
**colour_space** rgb, hsv, hls, ycrcb, lab or luv<br>
**num_clusters** integer between 1 and 100, e.g. 5 (required with ward and kmeans)<br>
**quantile** a float between 0 and 1, e.g. 0.01 (required with meanshift)

Ward is interesting, because it's configured with connectivity constraints - it'll cluster colours together by region. It's also the slowest algorithm, so may timeout after 25 seconds.

### Deployment

TODO

### Notes

Configure Redis to evict the less recently used keys first:

```
heroku plugins:install heroku-redis
heroku redis:maxmemory --policy allkeys-lru
```
