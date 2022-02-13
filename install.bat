python -m pip install virtualenv
virtualenv venv
venv\Scripts\activate.bat

python -m pip install -r requirements.txt && \
    jupyter nbextension enable --py widgetsnbextension
