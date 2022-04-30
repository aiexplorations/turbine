install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
lint:
	pylint --disable=R,C src/*.py

test:
	python -m pytest -vv --cov=frontend test_frontend.py
