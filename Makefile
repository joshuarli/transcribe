test:
	python -m pytest --parallel -m 'not integration'

pc:
	prek --quiet run --all-files

setup:
	prek install --install-hooks

