function active_subchirps = index_mapper(index_bits, n_c, k, g)
    % Maps index bits to active subchirp indices
    % Inputs:
    %   index_bits: Binary vector (1x104)
    %   n_c: Subchirps per sub-block (16)
    %   k: Active subchirps per sub-block (8)
    %   g: Number of sub-blocks (8)
    %
    % Output:
    %   active_subchirps: 1x64 indices

    active_subchirps = [];
    bits_per_block = floor(log2(nchoosek(n_c, k))); % 13 bits/block
    
    for beta = 1:g
        % Extract index bits for this sub-block
        start_idx = (beta-1)*bits_per_block + 1;
        end_idx = beta*bits_per_block;
        block_bits = index_bits(start_idx:end_idx);
        
        % Convert bits to decimal index
        dec_index = bi2de(block_bits, 'left-msb') + 1; % MATLAB uses 1-based indexing
        
        % Map to active subchirps using a lookup table (nchoosek)
        all_combinations = nchoosek(1:n_c, k);
        selected_indices = all_combinations(dec_index, :);
        
        % Convert to global indices
        block_start = (beta-1)*n_c + 1;
        global_indices = block_start + selected_indices - 1;
        
        active_subchirps = [active_subchirps, global_indices];
    end
end