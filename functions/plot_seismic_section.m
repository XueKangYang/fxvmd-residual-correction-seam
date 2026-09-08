function fig = plot_seismic_section(data, xAxis, tAxis, climRange, figTitle, outFile, figDPI)
%PLOT_SEISMIC_SECTION Plot and optionally save a 2-D seismic section.

    if nargin < 7 || isempty(figDPI)
        figDPI = 600;
    end

    fig = figure('Color','w','Name',figTitle);
    imagesc(xAxis, tAxis, data);
    set(gca,'YDir','reverse');
    axis tight;
    colormap(jet);
    colorbar;
    if ~isempty(climRange)
        caxis(climRange);
    end
    xlabel('Inline');
    ylabel('Time (s)');
    title(figTitle, 'Interpreter','none');

    if nargin >= 6 && ~isempty(outFile)
        outDir = fileparts(outFile);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        exportgraphics(fig, outFile, 'Resolution', figDPI);
    end
end
