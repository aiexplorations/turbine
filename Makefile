install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
lint:
	pylint --disable=R,C frontend.py

# Commented out the tests since this code is currently not covered by tests
# test:
#	python -m pytest -vv --cov=frontend test_frontend.py
