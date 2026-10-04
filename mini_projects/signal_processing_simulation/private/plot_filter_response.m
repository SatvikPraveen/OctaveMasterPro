% Location: mini_projects/signal_processing_simulation/private/plot_filter_response.m
% (split out of filter_design.m so it can be called from other files)

function plot_filter_response(b, a, fs)
    % Plot frequency response of filter
    
    [h, w] = freqz(b, a, 1024, fs);
    
    figure;
    subplot(2, 1, 1);
    semilogx(w, 20*log10(abs(h)));
    title('Filter Magnitude Response');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    
    subplot(2, 1, 2);
    semilogx(w, angle(h)*180/pi);
    title('Filter Phase Response');
    xlabel('Frequency (Hz)');
    ylabel('Phase (degrees)');
    grid on;
end
