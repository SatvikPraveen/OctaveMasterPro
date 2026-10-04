% Location: mini_projects/parallel_image_batch_processing/private/padarray.m
% Pad an array (Octave-core fallback for the image package's padarray).
%
%   B = padarray(A, padsize)                    pad with zeros on both sides
%   B = padarray(A, padsize, padval)            padval: scalar, 'replicate',
%                                               'symmetric' or 'circular'
%   B = padarray(A, padsize, padval, direction) direction: 'both' (default),
%                                               'pre' or 'post'
%
% padsize gives the amount of padding along each dimension; missing
% trailing dimensions are not padded.  Only what the demos need is
% implemented, but it matches the image package for these cases.

function B = padarray(A, padsize, padval, direction)
    if nargin < 3 || isempty(padval)
        padval = 0;
    end
    if nargin < 4 || isempty(direction)
        direction = 'both';
    end
    nd = max(ndims(A), numel(padsize));
    padsize = [padsize(:)', zeros(1, nd - numel(padsize))];
    switch lower(direction)
        case 'both'
            pre = padsize;  post = padsize;
        case 'pre'
            pre = padsize;  post = zeros(1, nd);
        case 'post'
            pre = zeros(1, nd);  post = padsize;
        otherwise
            error('padarray: unknown direction ''%s''', direction);
    end

    sz = size(A);
    sz(end+1:nd) = 1;
    if ischar(padval)
        % Index-based padding: build an index vector for each dimension
        idx = cell(1, nd);
        for d = 1:nd
            n = sz(d);
            k = (1 - pre(d)):(n + post(d));
            switch lower(padval)
                case 'replicate'
                    k = min(max(k, 1), n);
                case 'symmetric'
                    k = mod(k - 1, 2 * n);
                    k = min(k, 2 * n - 1 - k) + 1;
                case 'circular'
                    k = mod(k - 1, n) + 1;
                otherwise
                    error('padarray: unknown padding method ''%s''', padval);
            end
            idx{d} = k;
        end
        B = A(idx{:});
    else
        B = repmat(cast(padval, class(A)), sz + pre + post);
        idx = cell(1, nd);
        for d = 1:nd
            idx{d} = pre(d) + (1:sz(d));
        end
        B(idx{:}) = A;
    end
end
