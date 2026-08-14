%% Constants
% KS4 constants
template_length = 61; 
template_offset = 20;

% User-defined
filter = true;      % Apply filter before BOTM?
fc = 300;           % Filter cut-off frequency
t_max = 60;         % Max amount of time to compute noise covariance
                    % using (s)

BOTM_channels = 5;  % Channel span of BOTM filter

%% Add Useful Code to Path
addpath(genpath("util\"));

%% Ask User for Data Directory
% Kilosort directory
disp("Select Kilosort output directory.")
ks_file_path = uigetdir();

%% Ask User for Output Directory
disp("Select output directory.")
output_file_path = uigetdir();

%% Load Sorting Results
% Identified clusters and categories
sp = loadKSdir(ks_file_path);
cl_ids = sp.cids;
cl_g = sp.cgs;

num_ch = sp.n_channels_dat;
trodeNum = 1:num_ch;

% Raw data
s = split(sp.dat_path, ".");                  % Remove
np_file_name = char(join(s(1:end-1), "."));   % extension from
clear s;                                      % filename

% Spike times (samples)
spike_t = readNPY(fullfile(ks_file_path, 'spike_times.npy'));

% Spike identities
spike_cl = readNPY(fullfile(ks_file_path, 'spike_clusters.npy'));

%% Remove Spikes at Negative Times
spike_cl(spike_t < 0) = [];
spike_t(spike_t < 0) = [];

%% Load Neuropixels Data
% Load the meta file
fid = fopen(strcat(np_file_name, '.meta'));
C = textscan(fid, '%[^=] = %[^\r\n]');
fclose(fid);

% New empty struct
npMeta = struct();

% Convert each cell entry into a struct entry
for i = 1:length(C{1})
    tag = C{1}{i};
    if tag(1) == '~'
        % remake tag excluding first character
        tag = sprintf('%s', tag(2:end));
    end
    npMeta.(tag) = C{2}{i};
end

np_fs = str2double(npMeta.imSampRate); % NPx sample rate (Hz)

np_ch = eval("[" + npMeta.snsSaveChanSubset + "]");

% Find maximum time to load (last spike time + 1s)
Tmax = ceil(spike_t(end) / np_fs) + 1;

% Memory map the raw data
nChan = str2double(npMeta.nSavedChans);
nFileSamp = str2double(npMeta.fileSizeBytes) / ...
    (2 * nChan);
sizeA = [nChan, nFileSamp];
m = memmapfile(strcat(np_file_name, '.bin'), ...
    'Format', ...
    {'int16', sizeA, 'x'});

% Define the matrix of NPx data
neural = double(m.Data.x(trodeNum, ...
    1:min(Tmax * np_fs, nFileSamp))).';

%% Remove Sync Channel if Present
% Check if there is a sync channel
snsApLfSy = split(npMeta.snsApLfSy, ',');
num_sync_ch = str2double(snsApLfSy(end));

% If there are any sync channels, 
if num_sync_ch
    sync_chs = zeros([num_sync_ch, 1]);
    ch_i = 1;

    % Split the channels
    snsChanMap = split(npMeta.snsChanMap, ")(");
    snsChanMap(1) = erase(snsChanMap(1), '(');
    snsChanMap(end) = erase(snsChanMap(end), ')');

    % Look for any channels labelled as sync, and note their indices
    for i = 1:length(snsChanMap)
        map = split(snsChanMap(i), ':');
        if contains(map(1), "SY")
            sync_chs(ch_i) = str2double(map(end));
        end
    end
    
    % Delete the sync channels from the data
    neural(:, trodeNum == sync_chs) = [];
end

%% Filter NPx Data
% As the template is 61 samples long, the minimum frequency it can contain
% is ~500Hz. Therefore, we should filter the data to at least above 500Hz.
if filter
    [b,a] = butter(2, 2 * fc / np_fs, 'high');
    neural = filtfilt(b, a, neural);
end

%% Get Mean Waveform
% Compute
spike_mean = find_templates(neural, spike_t, spike_cl, cl_ids, ...
    template_length, template_offset);

% Plot
figure;
plot_waveforms(spike_mean, cl_ids);
sgtitle('Mean Waveforms');

%% Compute Covariance Matrix
% Compute
C = covariance_matrix(neural, spike_t, template_length, ...
    template_offset, ceil(t_max * np_fs));

% Plot
figure;
imagesc(C); colorbar;
title('Covariance Matrix')

%% Invert Covariance Matrix
% Compute
c = cond(C);
if c < 1e4
    warndlg("The condition number of the covariance matrix is less " + ...
        "than 10 000. This may lead to problems inverting it.");
end
alpha = 1;
W = inv(alpha * C + (1 - alpha) * eye(size(C)));

% Plot
figure;
imagesc(W); colorbar;
title('Whitening Matrix');

%% Compute Optimal Filter
% Compute
f = zeros(size(spike_mean));
for i = 1:length(cl_ids)
    f_v = W * reshape(spike_mean(i, :, :), [], 1);
    f(i, :, :) = reshape(f_v, size(spike_mean(i, :, :)));
end

% Plot
figure;
plot_waveforms(f, cl_ids);
sgtitle('Full BOTM Filters');

%% Restrict Filter to Sub-Channels 
% Compute
for i = 1:length(cl_ids)
    pows = sum(squeeze(spike_mean(i, :, :).^2), 1);
    [~, I] = sort(pows, 2, "descend");
    f(i, :, I(BOTM_channels+1:end)) = 0;
end

% Plot
figure;
plot_waveforms(f, cl_ids);
sgtitle('Restricted BOTM Filters');

%% Perform Filtering
D = zeros([size(neural, 1), length(cl_ids)]);

th = zeros([length(cl_ids), 1]);

for i = 1:length(cl_ids)
    
    for j = 1:size(neural, 2)
        D(:, i) = D(:, i) + conv(neural(:, j), ...
                squeeze(fliplr(squeeze(f(i, :, j)))), ...
                "same");
    end
    
    th(i) = 0.5 * sum(f(i, :, :) .* spike_mean(i, :, :), 'all') ...
        - log(numel(spike_t(spike_cl == cl_ids(i))) / size(D,1)) + ...
        log(1 - numel(spike_t) / size(D, 1));
end

%% Get GT and Sorted Spike Trains for Comparison
% Get "ground truth" spike trains
gtst = zeros(size(D));
for i = 1:length(cl_ids)
    gtst(spike_t(spike_cl == cl_ids(i))+1, i) = 1;
end

% Get initial sorted spike trains
sst = zeros(size(gtst));
scores = zeros(size(gtst));
for i = 1:length(cl_ids)
    [pks, ids] = findpeaks(D(:, i));
    sst(ids, i) = 1;

    scores(:, i) = ones([size(scores, 1), 1]) * ...
        2 * min(pks, [], 'all');
    scores(ids, i) = pks;
end

% Align trains
for i = 1:length(cl_ids)
    [c, lags] = xcorr(sst(:, i), gtst(:, i));
    [~, max_loc] = max(c);
    shift = lags(max_loc);

    sst_padded = [zeros([abs(shift), 1]); sst(:, i); zeros([abs(shift), 1])];
    sst_padded = circshift(sst_padded, -shift, 1);
    sst(:, i) = sst_padded(abs(shift)+1:end-abs(shift));

    scores_padded = [zeros([abs(shift), 1]); scores(:, i); zeros([abs(shift), 1])];
    scores_padded = circshift(scores_padded, -shift, 1);
    scores(:, i) = scores_padded(abs(shift)+1:end-abs(shift));
end

%% Allow For a Little Bit of Jitter
% To do this, we max pool the gtst and sst
w = 15; % 0.5ms window

gtst_pooled = zeros([floor(size(gtst, 1) / w), size(gtst, 2)]);
scores_pooled = zeros([floor(size(scores, 1) / w), size(scores, 2)]);

for j = 1:size(gtst_pooled, 2)
    for i = 1:size(gtst_pooled, 1)-1
        gtst_pooled(i, j) = max(gtst((1:w) + (i-1) * w, j), [], 'all');
        scores_pooled(i, j) = max(scores((1:w) + (i-1) * w, j), [], 'all');
    end
    gtst_pooled(end, j) = max(gtst(w * i:end, j), [], 'all');
    scores_pooled(end, j) = max(scores(w * i:end, j), [], 'all');
end


%% Plot ROC Curves for each Unit
figure;
for i = 1:length(cl_ids)
    rocObj = rocmetrics(gtst_pooled(:, i), ...
        scores_pooled(:, i) - th(i), 1, ...
    AdditionalMetrics = ["Accuracy", ...
    "TrueNegativeRate", ...
    "PositivePredictiveValue", ...
    "NegativePredictiveValue"]);

    plot(rocObj);
    title(cl_ids(i));
    pause;
end

%% Save Outputs
for i = 1:length(cl_ids)

    % Get the filter
    sub_f = squeeze(f(i, :, :));

    % Save channels for filter
    filter_channels_id = any(sub_f, 1);
    filter_channels = np_ch(filter_channels_id);

    % Print the channel list for each filter
    disp("Cluster " + cl_ids(i) + " channels: " + ...
        join(split(int2str(filter_channels), '  '), ','));

    fid = fopen(fullfile(output_file_path, ...
        strcat("channels_", int2str(cl_ids(i)), ".bin")), "w+");
    fwrite(fid, filter_channels, 'int32');
    fclose(fid);

    % Save filter
    sub_f_reduced = sub_f(:, filter_channels_id);
    fid = fopen(fullfile(output_file_path, ...
        strcat("filter_", int2str(cl_ids(i)), ".bin")), "w+");
    fwrite(fid, sub_f_reduced, 'float64');
    fclose(fid);    
    
    % Save threshold
    fid = fopen(fullfile(output_file_path, ...
        strcat("threshold_", int2str(cl_ids(i)), ".bin")), "w+");
    fwrite(fid, th(i), 'float64');
    fclose(fid);
end