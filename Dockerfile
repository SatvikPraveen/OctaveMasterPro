# Reproducible GNU Octave + JupyterLab environment for OctaveMasterPro.
#
# Ubuntu 24.04 ships GNU Octave 8.4, the version the test suite and the
# published flagship results were produced with.  Python tooling is pinned.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        octave \
        octave-signal \
        octave-statistics \
        octave-control \
        octave-image \
        octave-io \
        octave-parallel \
        gnuplot-nox \
        ghostscript \
        fonts-freefont-otf \
        python3 \
        python3-venv \
        git \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated Python environment for Jupyter (PEP 668 forbids system pip).
RUN python3 -m venv /opt/jupyter \
    && /opt/jupyter/bin/pip install \
        "jupyterlab==4.6.4" \
        "octave_kernel==1.1.1"
ENV PATH="/opt/jupyter/bin:${PATH}"

# Run as an unprivileged user.
RUN useradd --create-home --shell /bin/bash researcher
USER researcher
WORKDIR /home/researcher/OctaveMasterPro

RUN python3 -m octave_kernel install --user

# Put the library on Octave's path in every session.
RUN printf '%s\n' \
      "addpath ('/home/researcher/OctaveMasterPro/inst');" \
      "addpath ('/home/researcher/OctaveMasterPro/utils');" \
    > /home/researcher/.octaverc

COPY --chown=researcher:researcher . /home/researcher/OctaveMasterPro

EXPOSE 8888

# Jupyter requires a token.  Set JUPYTER_TOKEN at run time (see
# docker-compose.yml); if unset, Jupyter generates one and prints it.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
