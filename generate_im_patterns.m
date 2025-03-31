function patterns = generate_im_patterns(Ns_comm, num_classes)
    % Input Validation
    validateattributes(num_classes, {'numeric'}, {'scalar', 'integer', 'positive'});
    
    patterns = cell(1, num_classes); % Now safe
    subchirps_per_class = 48;

    % Class 1: Random 48 subchirps
    patterns{1} = randperm(Ns_comm, subchirps_per_class);
    
    % Class 2: Comb pattern (every 2nd subchirp)
    comb_indices = 1:2:Ns_comm;
    patterns{2} = comb_indices(1:subchirps_per_class);
    
    % Class 3: Block pattern (first 24 + last 24)
    patterns{3} = [1:24, (Ns_comm-23):Ns_comm];
    
    % Class 4: Fill remaining subchirps
    all_used = unique([patterns{1:3}]);
    remaining = setdiff(1:Ns_comm, all_used);
    needed = max(subchirps_per_class - length(remaining), 0);
    
    if needed > 0
        extra = randperm(length(all_used), needed);
        patterns{4} = [remaining, all_used(extra)];
    else
        patterns{4} = remaining;
    end
    
    % Force exactly 48 subchirps
    patterns{4} = patterns{4}(1:subchirps_per_class);
    
    % Final validation
    final_used = unique([patterns{:}]);
    assert(length(final_used) == Ns_comm, 'Full coverage failed');
end