function plot_waveforms(w, ids)

min_v = min(w, [], 'all');
max_v = max(w, [], 'all');

% Plot result
tiledlayout(1, length(ids), "TileSpacing", "tight");
for i = 1:length(ids)
    nexttile;
    imagesc(squeeze(w(i, :, :)).');
    axis off;
    title(ids(i));
    clim([min_v max_v]);
    colormap gray;
end

end