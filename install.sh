#!/usr/bin/env bash

python -m pip install virtualenv
virtualenv venv
source venv/bin/activate

python -m pip install -r requirements.txt && \
    jupyter nbextension enable --py widgetsnbextension
