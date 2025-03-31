function stft_cell = process_stft_batch(X, num_frames, num_features)
    num_samples = size(X, 2);
    stft_cell = cell(num_samples, 1);

    parfor i = 1:num_samples
        stft_matrix = compute_stft(X(:,i), [64 128 256], 256);
        
        % No need to transpose - compute_stft now handles it
        stft_cell{i} = stft_matrix;
        
        
        % Update assertion check
        assert(size(stft_cell{i}, 1) == num_frames && ...
               size(stft_cell{i}, 2) == num_features, ...
               'STFT dimensions mismatch at sample %d: got [%d×%d], expected [%d×%d]', ...
               i, size(stft_cell{i},1), size(stft_cell{i},2), num_frames, num_features);
    end
end



% function stft_cell = process_stft_batch(X, num_frames, num_features)
%     % PROCESS_STFT_BATCH - Converts STFT matrices to LSTM-compatible cell arrays
%     %
%     % Inputs:
%     %   X            : [Nc × num_samples] raw signal matrix
%     %   num_frames   : Number of time frames (from compute_stft)
%     %   num_features : Number of features per frame (from compute_stft)
%     %
%     % Output:
%     %   stft_cell    : {num_samples × 1} cell array of [num_frames × num_features] matrices
% 
%     num_samples = size(X, 2);
%     stft_cell = cell(num_samples, 1);
% 
%     parfor i = 1:num_samples
%         % Compute STFT for each signal
%         stft_matrix = compute_stft(X(:,i), [64 128 256], 256);
% 
%         % Transpose to [num_frames × num_features]
%         % stft_cell{i} = stft_matrix';
% 
%         stft_cell{i} = stft_matrix
% 
%         % Validate dimensions
%         assert(size(stft_cell{i}, 1) == num_frames && ...
%                size(stft_cell{i}, 2) == num_features, ...
%                'STFT dimensions mismatch at sample %d', i);
%     end
% end