lint:
	ruff check .

test:
	pytest -q

ci: lint test
