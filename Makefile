.PHONY: test audit reproduce archive check

test:
	PYTHONPATH=src python -m pytest

audit:
	PYTHONPATH=src python -m metacog.cli.audit .

reproduce:
	PYTHONPATH=src MPLCONFIGDIR=/tmp python -m metacog.cli.reproduce \
		--input artifacts/publication --data data --output outputs/paper

archive:
	mkdir -p outputs
	git archive --format=zip --output=outputs/release-code.zip HEAD

check: test audit reproduce
