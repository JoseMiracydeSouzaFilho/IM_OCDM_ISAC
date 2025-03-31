function [stft_matrix, num_frames, num_features] = compute_stft(signal, win_sizes, nfft)
    % Inputs:
    %   signal    : Input signal [Nc × 1]
    %   win_sizes : Array of window sizes (e.g., [64, 128, 256])
    %   nfft      : FFT size
    % Outputs:
    %   stft_matrix : Concatenated STFT features [num_features × num_frames]
    %   num_frames  : Number of time frames (standardized)
    %   num_features: Total features (win_sizes × nfft)

    % Initialize
    stft_cells = cell(1, length(win_sizes));
    max_frames = 0;
    
    % First pass: Compute STFT for each window and find maximum frames
    for k = 1:length(win_sizes)
        win = win_sizes(k);
        noverlap = round(win * 0.75); % 75% overlap
        [~, ~, ~, frames] = spectrogram(signal, win, noverlap, nfft);
        stft_cells{k} = abs(frames);
        max_frames = max(max_frames, size(frames, 2));
    end
    
    % Pad/Truncate all STFTs to max_frames
    for k = 1:length(win_sizes)
        current = stft_cells{k};
        current_frames = size(current, 2);
        if current_frames < max_frames
            % Pad with zeros
            stft_cells{k} = [current, zeros(size(current, 1), max_frames - current_frames)];
        else
            % Truncate to max_frames
            stft_cells{k} = current(:, 1:max_frames);
        end
    end
    
    % Vertical concatenation
   % Transpose during vertical concatenation
    stft_matrix = vertcat(stft_cells{:})';  % Note the transpose here
    num_features = size(stft_matrix, 2);    % Now columns are features
    num_frames = size(stft_matrix, 1);      % Rows are time frames
end