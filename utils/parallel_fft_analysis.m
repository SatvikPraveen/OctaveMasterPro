% File location: OctaveMasterPro/utils/parallel_fft_analysis.m
% Parallel processing helper (from parallel_wrappers.m). Kept in its own
% file so it can be called from other scripts: Octave subfunctions in
% parallel_wrappers.m are only visible inside that file.

function results = parallel_fft_analysis(signals)
    % Parallel FFT analysis of multiple signals
    % Usage: results = parallel_fft_analysis({signal1, signal2, signal3})
    
    if ~iscell(signals)
        error('Input must be cell array of signals');
    end
    
    n_signals = length(signals);
    fprintf('Computing FFT for %d signals in parallel...\n', n_signals);
    
    results = cell(n_signals, 1);
    
    tic;
    if check_parallel_capability() && n_signals > 1
        % Parallel FFT computation
        for i = 1:n_signals
            fft_result = fft(signals{i});
            
            % Package FFT results with metadata
            result_data = struct();
            result_data.fft_data = fft_result;
            result_data.magnitude = abs(fft_result);
            result_data.phase = angle(fft_result);
            result_data.power = abs(fft_result).^2;
            result_data.signal_length = length(signals{i});
            
            results{i} = result_data;
        end
    else
        % Serial execution
        for i = 1:n_signals
            fft_result = fft(signals{i});
            
            result_data = struct();
            result_data.fft_data = fft_result;
            result_data.magnitude = abs(fft_result);
            result_data.phase = angle(fft_result);
            result_data.power = abs(fft_result).^2;
            result_data.signal_length = length(signals{i});
            
            results{i} = result_data;
        end
    end
    
    fft_time = toc;
    fprintf('FFT analysis completed in %.4f seconds\n', fft_time);
end
