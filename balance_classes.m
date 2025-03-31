% Ensure all classes (index modulation patterns) 
% have the same number of samples to prevent the LSTM from biasing 
% toward majority classes.

% counts: Number of samples per class
% edges: Class labels (e.g., [1, 2, 3, 4])
% max_count: Largest class size

function [Y_balanced, X_balanced] = balance_classes(Y, X, num_classes)
    % Ensure Y is categorical and convert to column vector
    if ~iscategorical(Y)
        Y = categorical(Y, 1:num_classes);
    end
    Y = Y(:); % Force column vector
    
    % Get class counts and categories
    [counts, ~] = histcounts(Y);
    class_names = categories(Y);
    max_count = max(counts);
    
    % Initialize balanced datasets
    X_balanced = X;
    Y_balanced = Y; % Now a column vector
    
    for c = 1:num_classes
        current_count = counts(c);
        if current_count < max_count
            needed = max_count - current_count;
            
            % Get class samples
            class_samples = X(:, Y == class_names{c});
            
            % Check for empty classes
            if isempty(class_samples)
                error('Class %s has no samples!', class_names{c});
            end
            
            % Duplicate with noise
            synth_indices = randi(size(class_samples, 2), 1, needed);
            synth_samples = class_samples(:, synth_indices) + 0.02*randn(size(X,1), needed);
            
            % Create matching categorical labels (column vector)
            new_labels = categorical(repelem(class_names(c), needed), class_names);
            new_labels = new_labels(:); % Ensure column
            
            % Append
            X_balanced = [X_balanced synth_samples];
            Y_balanced = [Y_balanced; new_labels]; 
        end
    end
end