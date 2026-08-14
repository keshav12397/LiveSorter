function C = covariance_matrix(data, spike_t, template_length, ...
    template_offset, s_max)

% Reduce data matrix size if desired
if s_max < size(data, 1)
    data = data(1:s_max, :);
end

% Ensure correct types
template_length = cast(template_length, 'int64');
template_offset = cast(template_offset, 'int64');

% Find periods of data without any spikes
spike_present = zeros([size(data, 1), 1]);

% Treat only periods with no spikes from any neurons as noise
for i = 1:length(spike_t)

    sp_start = spike_t(i) - template_offset;
    sp_end = spike_t(i) + template_length - 1 + template_offset;

    if sp_start < 1
        sp_start = 1;
    end

    if sp_end > length(spike_present)
        sp_end = length(spike_present);
    end

    spike_present(sp_start:sp_end) = 1;
end

% Find the location of gaps in the data
gap_starts = find(diff(spike_present) == -1);
gap_ends = find(diff(spike_present) == 1);

% Correct for first gap having no start
if (gap_ends(1) < gap_starts(1))
    gap_starts = [1; gap_starts];
end

% Correct for last gap having no end
if (gap_starts(end) > gap_ends(end))
    gap_ends = [gap_ends; size(data, 1)];
end

% Find the lengths of the gaps
gap_lengths = gap_ends - gap_starts;

% Find only those gaps that are longer than twice the template
long_gap_ids = gap_lengths >= 2 * template_length;
gap_starts = gap_starts(long_gap_ids);
gap_ends = gap_ends(long_gap_ids);
gap_lengths = gap_lengths(long_gap_ids);
total_gap_length = sum(gap_lengths, 'all');

% Initialize covariance matrix
C = zeros(template_length * size(data, 2));

% Compute covariance matrix
for i = 1:size(data, 2)
    for k = 1:i       
        % Initialize the combined cov thing
        cov_combined = zeros([2 * template_length - 1, 1]);

        % Initialize this Toeplitz thing
        for g = 1:length(gap_lengths)

            t = xcov(data(gap_starts(g):gap_ends(g), i), ...
                data(gap_starts(g):gap_ends(g), k), ...
                template_length - 1, 'biased');

            cov_combined = cov_combined + gap_lengths(g) * t;

        end

        cov_combined = cov_combined / total_gap_length;

        T = toeplitz(flipud(cov_combined(1:template_length)), ...
            cov_combined(template_length:end));

        % Need to really check if this is right (esp wrt tranposes)
        C((1:template_length) + (i-1) * template_length, ...
            (1:template_length) + (k-1) * template_length) = T;

        if (i ~= k)
            C((1:template_length) + (k-1) * template_length, ...
                (1:template_length) + (i-1) * template_length) = T.';
        end
    end
end


end