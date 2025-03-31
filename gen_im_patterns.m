function [active_subchirps, radar_subchirps] = gen_im_patterns(Nc_comm, g, k)
    % Inputs:
    % função intiva 
    %   Nc_comm = Subchirps allocated for communication (128)
    %   g = Number of sub-blocks (8)
    %   k = Active subchirps per sub-block (8)
    
    n_c = Nc_comm / g; % Subchirps per sub-block (16)
    active_subchirps = [];
    
    for beta = 1:g
        % Randomly select k active subchirps in each sub-block
        block_start = (beta-1)*n_c + 1;
        block_indices = block_start : block_start + n_c - 1;
        active_in_block = block_indices(randperm(n_c, k));
        active_subchirps = [active_subchirps, active_in_block];
    end
    
    % Radar subchirps (remaining indices)
    radar_subchirps = setdiff(1:256, active_subchirps);
end