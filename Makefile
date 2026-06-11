.PHONY: help install install-full test smoke services-up services-down silver

help:
	@echo "Signpost artifact targets:"
	@echo "  install       - install offline-test deps only (pip install -e '.[test]')"
	@echo "  install-full  - install the full-pipeline deps (requirements.txt)"
	@echo "  test          - run the offline test suites (no ES / LLM / corpus)"
	@echo "  smoke         - alias for test (zero-setup reviewer check)"
	@echo "  services-up   - docker compose up -d (Postgres / Valkey / MinIO)"
	@echo "  services-down - docker compose down"
	@echo "  silver        - build silver-evidence targets (needs ECNU_API_* env)"
	@echo ""
	@echo "Pass extra args via ARGS=... (e.g. make silver ARGS='--limit 10')"

install:
	pip install -e '.[test]'

install-full:
	pip install -r requirements.txt

test smoke:
	python -m pytest tests/test_sketch_chaining.py tests/test_stats_ci.py \
	    tests/test_iso_call_baseline.py tests/test_silver_builder.py -q

services-up:
	docker compose -f docker/docker-compose.yml up -d

services-down:
	docker compose -f docker/docker-compose.yml down

silver:
	python scripts/build_silver_evidence.py $(ARGS)
