# Signal Processing Simulation

**Location:** `mini_projects/signal_processing_simulation/README.md`

## Overview

This mini-project demonstrates fundamental signal processing concepts using Octave, including signal generation, filtering, and spectral analysis.

## Features

- Generate various signal types (sine, square, sawtooth, noise)
- Design and apply digital filters (lowpass, highpass, bandpass)
- Perform FFT analysis and spectral visualization
- Real-time signal manipulation and analysis

## Files

- `signal_generator.m` - Signal generation functions
- `filter_design.m` - Digital filter design and application
- `spectrum_analyzer.m` - FFT analysis and plotting
- `signal_demo.m` - Main demonstration script

## Usage

```octave
# Run the main demonstration
signal_demo

# Generate specific signals
[t, signal] = generate_signal('sine', 1000, 50, 1, 0.1);

# Apply filters
filtered = apply_lowpass(signal, 0.3);

# Analyze spectrum
spectrum_analyzer(signal, 1000);
```

## Requirements

- Octave with signal processing package
- Plotting capabilities (gnuplot)

## Sample Outputs

- Time domain signal plots
- Frequency domain analysis
- Filter response characteristics
- Before/after filtering comparisons
