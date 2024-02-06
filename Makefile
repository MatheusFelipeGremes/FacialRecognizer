.PHONY: coverage build-docs

coverage:
	coverage erase && coverage run -m pytest tests/ && coverage report
