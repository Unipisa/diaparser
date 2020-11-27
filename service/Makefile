# docs https://live.european-language-grid.eu/

HOST = 127.0.0.1
PORT = 8888

WORKERS = 4

start:
	export CUDA_VISIBLE_DEVICES=""; uvicorn server:app --host $(HOST) --port $(PORT) --workers $(WORKERS)
