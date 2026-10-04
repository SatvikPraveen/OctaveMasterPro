# Setup guide

OctaveMasterPro needs **GNU Octave ≥ 6.1**. The test suite and the
published results were produced with **Octave 8.4** on Ubuntu 24.04
(OpenBLAS 0.3.26, LAPACK 3.12.0); every result file records its own
environment in an `environment.txt` file.

## Option 1: Docker (fully pinned)

```bash
git clone https://github.com/SatvikPraveen/OctaveMasterPro.git
cd OctaveMasterPro
docker compose build

# JupyterLab on http://127.0.0.1:8888 (token required)
JUPYTER_TOKEN=$(openssl rand -hex 16) docker compose up jupyter
# then open http://127.0.0.1:8888/lab?token=<the token you set>

# Octave command line, or any make target, inside the container
docker compose run --rm octave-cli
docker compose run --rm octave-cli make check
```

The image runs as an unprivileged user, puts `inst/` and `utils/` on the
Octave path via `~/.octaverc`, and publishes Jupyter on localhost only.
The repository is bind-mounted, so edits on the host are visible inside
the container.

## Option 2: Native install

**Ubuntu / Debian**

```bash
sudo apt-get install octave octave-signal octave-statistics gnuplot-nox make
```

**macOS (Homebrew)**

```bash
brew install octave gnuplot
octave --eval 'pkg install -forge signal statistics'
```

**Windows**: install Octave from <https://octave.org/download>, then run
the `octave` commands below from the Octave prompt (`make` is optional).

Then, from the repository root:

```bash
make check               # 125+ library tests and a parse check of every .m file
make env                 # print Octave / BLAS / LAPACK / package versions
```

`signal` and `statistics` are optional. Library code depends only on
core Octave; one test compares against `pwelch` when `signal` is installed.

### Jupyter (optional)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "jupyterlab==4.6.4" "octave_kernel==1.1.1"
python -m octave_kernel install --user
jupyter lab
```

## Using the library

Either add it to the path for a session:

```octave
addpath ("/path/to/OctaveMasterPro/inst");
omp.repro.env_info ("print");
```

or install it as an Octave package:

```bash
make package             # builds build/octavemasterpro.tar.gz from HEAD
octave --eval 'pkg install build/octavemasterpro.tar.gz'
```

```octave
pkg load octavemasterpro
help omp.stats.bootstrap_ci
```

## Reproducing the published results

```bash
make audit               # flagship_project/results/data_audit.md
make experiment          # flagship_project/results/* (5 seeds; tens of minutes)
octave --eval 'addpath experiments; qr_orthogonality; rsvd_accuracy'
```

Every function that draws random numbers accepts a `"Seed"` option or is
seeded explicitly. Results should match bit-for-bit on the same Octave
and BLAS build. On other BLAS builds, expect agreement to within
floating-point reassociation; it will not be bitwise.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md). Common issues:

| Symptom | Cause / fix |
|---|---|
| `'omp' undefined` | `inst/` is not on the path: `addpath inst` or `pkg load octavemasterpro`. |
| `parse error` inside `[...]` or `{...}` | In matrix and cell literals, `f (x)` with a space is two elements. Write `f(x)`. |
| Figures fail headless | Use `graphics_toolkit ("gnuplot")` and `figure ("visible", "off")`, or run with `--no-window-system`. |
| `readtable` undefined | Octave has no tables; use `omp.io.read_csv`. |
