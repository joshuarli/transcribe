test:
	python -m pytest --parallel -m 'not integration'

pc:
	prek --quiet run --all-files

setup:
	prek install --install-hooks
	python -m spacy download en_core_web_trf

