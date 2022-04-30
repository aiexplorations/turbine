install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
lint:
	pylint --disable=R,C src/*.py

test:
	python -m unittest discover .
	python -m pytest -vv --cov=src/database tests/test_database.py
