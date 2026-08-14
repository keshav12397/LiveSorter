function T = find_templates(data, spike_times, spike_cl, cl_ids, ...
    template_length, template_offset)
%FIND_TEMPLATES  Compute mean waveform of spike clusters

% Define constants
num_cl = length(cl_ids);  % Number of clusters
num_ch = size(data, 2);   % Number of channels

% Ensure correct types
template_length = cast(template_length, 'int64');
template_offset = cast(template_offset, 'int64');

% Initialize T
T = zeros([num_cl, template_length, num_ch]);

% Compute mean spike
for i = 1:num_cl
    t = spike_times(spike_cl == cl_ids(i));

    % TODO: Change next step to not compute twice
    t = t(t > template_offset);
    t = t(t < size(data, 1) - template_length + template_offset);

    for j = 1:length(t)
        T(i, :, :) = squeeze(T(i, :, :)) + ...
            data((1:template_length) + t(j) - template_offset, :);
    end
    T(i, :, :) = T(i, :, :) / length(t);
end

end