# NanoPInv

`nanopinv` is a small Python library that takes inspiration from a draft toolbox for probablistic inversion in geostatistics made by [Thomas Mejer Hansen](https://www.au.dk/en/tmeha@geo.au.dk/).

It is spiritually inspired by the many tiny/nano educational libraries made for auto-differentiation.
As such, it aims to have a small, readable, and extensible source code rather than being highly performant and complete.

Additionally, it tries to be well-packaged and idiomatic.

It is made with the Master's Course [_30760 Inverse Problems and Machine Learning in Earth and Space Physics_](https://lifelonglearning.dtu.dk/en/space/single-course/inverse-problems-and-machine-learning-in-earth-and-space-physics/) at [Technical University of Denmark](https://www.dtu.dk/english/) in mind.

## Overview

- Leverages [`gstools`](https://geostat-framework.readthedocs.io/)' [covariance models](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/02_cov_model/index.html) for construction of covariance matrices.
  - Student's may find it interesting to implement their own covariance models
