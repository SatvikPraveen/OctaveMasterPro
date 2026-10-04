% Location: mini_projects/signal_processing_simulation/spectrum_analyzer.m
% FFT Analysis and Spectral Visualization Tools
%
% The analysis functions (analyze_spectrum, plot_spectrum_analysis,
% compare_windows, spectrogram_analysis, peak_detection_demo,
% demo_spectrum_analyzer) live in private/.
%
%   spectrum_analyzer(signal, fs, ...)   analyze and plot a signal's spectrum
%                                        (same options as analyze_spectrum)
%   [f, mag, ph] = spectrum_analyzer(signal, fs, ...)
%   spectrum_analyzer()                  run the spectrum analyzer demo

function varargout = spectrum_analyzer(varargin)
    if nargin == 0
        demo_spectrum_analyzer();
        return;
    end
    [frequencies, magnitude, phase] = analyze_spectrum(varargin{:});
    if nargout > 0
        varargout = {frequencies, magnitude, phase};
        varargout = varargout(1:nargout);
    end
end
