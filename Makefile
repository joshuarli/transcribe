audio:
	fd --threads $$(( $$(nproc) - 1 )) --no-ignore -e mp3 . cache -x sh -c 'echo "$$1 -> $${1%.mp3}.opus" && ffmpeg -loglevel error -threads 1 -i "$$1" -c:a libopus -b:a 48k -ac 1 -compression_level 0 "$${1%.mp3}.opus"' _ {}

pc:
	prek --quiet run --all-files

setup:
	prek install --install-hooks

clip-parakeet.json:
	uv run transcribe --backend parakeet clip.mp3 clip-parakeet.json --diarize

clip-mlx.json:
	uv run transcribe --backend mlx clip.mp3 clip-mlx.json --diarize
