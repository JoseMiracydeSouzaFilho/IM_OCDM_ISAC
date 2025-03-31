%% Sanity Test
clear;
clc;
test_pattern = generate_im_patterns(128, 4);
all_indices = [test_pattern{:}];
unique_count = length(unique(all_indices));

fprintf('Subchirps utilizados: %d/128\n', unique_count);
assert(unique_count >= 100, ...
    'Padrões IM usam apenas %d subchirps. Requerido: >=100', unique_count);

%% Diagnostic Tool: Visualize Activated Subchirps
figure('Position', [100 100 800 400]);

% Get all activated indices
activated = unique([test_pattern{:}]);
unactivated = setdiff(1:128, activated);

% Plot activation status
subplot(2,1,1);
stem(activated, ones(size(activated)), 'filled', 'MarkerSize', 3);
title('Activated Subchirps (1-128)');
xlabel('Subchirp Index'); ylim([0 1.2]);

subplot(2,1,2);
stem(unactivated, ones(size(unactivated)), 'r', 'filled', 'MarkerSize', 3);
title('Unactivated Subchirps');
xlabel('Subchirp Index'); ylim([0 1.2]);

fprintf('Activated: %d/128\nUnactivated: %d/128\n', ...
    length(activated), length(unactivated));

%% Check class uniqueness
for c = 1:4
    fprintf('Class %d: %d subchirps\n', c, length(test_pattern{c}));
end