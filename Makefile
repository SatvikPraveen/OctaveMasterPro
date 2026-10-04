# OctaveMasterPro developer tasks.  Requires GNU Octave >= 6 on PATH.
OCTAVE ?= octave --no-gui --no-window-system --quiet
EXP     = flagship_project/experiment

.PHONY: help test parse check docs-check experiment experiment-quick audit env package docker clean

help:
	@echo "make test              run the library test suite (inst/+omp)"
	@echo "make parse             parse-check every tracked .m file"
	@echo "make check             test + parse"
	@echo "make docs-check        execute every example in docs/usage_examples.md"
	@echo "make audit             audit the shipped flagship datasets"
	@echo "make experiment        full flagship simulation study (~10-20 min)"
	@echo "make experiment-quick  2-seed smoke run (~1 min) into /tmp"
	@echo "make env               print the numerical environment"
	@echo "make package           build an Octave package tarball (pkg install)"
	@echo "make docker            build the Jupyter/Octave Docker image"

test:
	$(OCTAVE) --eval "addpath tests; exit (! run_tests ());"

parse:
	$(OCTAVE) --eval "addpath tests; exit (! check_parse ());"

check: test parse

docs-check:
	$(OCTAVE) --eval "addpath docs; exit (! check_usage_examples ());"

audit:
	$(OCTAVE) --eval "addpath ('$(EXP)'); audit_shipped_data ();"

experiment:
	$(OCTAVE) --eval "addpath ('$(EXP)'); graphics_toolkit ('gnuplot'); run_pdm_experiment ();"

experiment-quick:
	$(OCTAVE) --eval "addpath ('$(EXP)'); run_pdm_experiment ('Quick', true, 'Figure', false, 'OutDir', tempname ());"

env:
	$(OCTAVE) --eval "addpath inst; omp.repro.env_info ('print');"

package:
	mkdir -p build
	git archive --format=tar.gz --prefix=octavemasterpro/ -o build/octavemasterpro.tar.gz HEAD \
	  DESCRIPTION COPYING inst
	@echo "Install with: octave --eval \"pkg install build/octavemasterpro.tar.gz\""

docker:
	docker build -t octavemasterpro:latest .

clean:
	rm -rf build octave-workspace
