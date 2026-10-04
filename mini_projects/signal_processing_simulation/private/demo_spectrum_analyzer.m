% Location: mini_projects/signal_processing_simulation/private/demo_spectrum_analyzer.m
% (split out of spectrum_analyzer.m so it can be called from other files)

function demo_spectrum_analyzer()
    % Comprehensive demonstration of spectrum analyzer
    
    fprintf('Spectrum Analyzer Demonstration\n');
    fprintf('===============================\n\n');
    
    % Generate test signals
    fs = 1000;
    t = 0:1/fs:5;
    
    % Multi-component signal
    signal1 = sin(2*pi*50*t) + 0.7*sin(2*pi*120*t) + 0.4*sin(2*pi*200*t) + 0.1*randn(size(t));
    
    % Chirp signal
    signal2 = chirp(t, 20, 5, 200) + 0.05*randn(size(t));
    
    % Analyze multi-component signal
    fprintf('1. Analyzing multi-component signal (50, 120, 200 Hz)...\n');
    [freq1, mag1, phase1] = analyze_spectrum(signal1, fs);
    
    fprintf('2. Comparing window functions...\n');
    compare_windows(signal1, fs);
    
    fprintf('3. Peak detection demonstration...\n');
    peak_detection_demo(signal1, fs);
    
    fprintf('4. Analyzing chirp signal...\n');
    analyze_spectrum(signal2, fs, 'window', 'blackman');
    
    fprintf('5. Spectrogram analysis of chirp...\n');
    spectrogram_analysis(signal2, fs, 'window_length', 128, 'overlap', 64);
    
    fprintf('\nSpectrum Analyzer demonstration complete!\n');
end
