% Location: mini_projects/parallel_image_batch_processing/private/get_memory_usage.m

function memory_mb = get_memory_usage()
    % Resident set size of this Octave process in MB, read from
    % /proc/self/status (Linux).  Returns NaN where that is unavailable:
    % a missing measurement must never be replaced by a made-up number.
    memory_mb = NaN;
    try
        txt = fileread('/proc/self/status');
        tok = regexp(txt, 'VmRSS:\s*(\d+)\s*kB', 'tokens', 'once');
        if ~isempty(tok)
            memory_mb = str2double(tok{1}) / 1024;
        end
    catch
        memory_mb = NaN;
    end
end
