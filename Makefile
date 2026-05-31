.PHONY: install test bench bench-hit bench-fanout bench-pareto clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

bench: bench-hit bench-fanout bench-pareto

bench-hit:
	python benchmarks/cache_hit_rate.py

bench-fanout:
	python benchmarks/fan_out_sweep.py

bench-pareto:
	python benchmarks/acceptance_hit_pareto.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
