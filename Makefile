install:
	# Required for pyaudio on linux
	apt install portaudio19-dev

shell:
	poetry shell

run:
	poetry run python -m src.main

lint:
	poetry run ruff check src --fix 
