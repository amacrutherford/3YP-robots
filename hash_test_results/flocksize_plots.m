%T = readtable('flock_size_data.xlsx');
T = readtable('3YP_data.xlsx','Sheet',1);

partial = 0.15; % percentage for partial failure
complete = 0.75; 

A = T.Variables;

R = zeros(4, width(T));
tick_names = {};

for col = 1 : width(T)
    name = T.Properties.VariableNames{col}(2:end);
    tick_names{col} = name;
    num = str2double(name);
    pthresh = ceil(partial*num);
    cthresh = floor(complete*num);
    R(1,col) = sum(T{:,col} < 1) / length(T{:,col});
    R(2,col) = sum(T{:,col} <= pthresh & T{:,col} > 0) / length(T{:,col});
    R(3,col) = sum(T{:,col} > pthresh & T{:,col} < cthresh) / length(T{:,col});
    R(4,col) = sum(T{:,col} >= cthresh) / length(T{:,col});
end

figure('Position', [50 50 600 550])
bar(R.'*100, 'stacked')
set(gca,'XTickLabel',tick_names);
xlabel('Flock size')
ytickformat('percentage')
ylabel('% of total')
ylim([0 100])
axis square

legend({'Success','Partial Failure', 'Serious Failure', 'Complete Failure'},'Location','northoutside', 'Orientation','horizontal')
f = gcf;
%exportgraphics(f,'barchart.png')

