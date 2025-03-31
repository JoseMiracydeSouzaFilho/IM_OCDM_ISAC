classdef attentionLayer < nnet.layer.Layer
    properties (Learnable)
        W_h  % [AttentionSize x HiddenStateDim]
        W_s  % [AttentionSize x QuerySize]
        v    % [1 x AttentionSize]
        b    % [AttentionSize x 1]
        s    % [QuerySize x 1] (learned query vector)
    end

    properties
        AttentionSize
        QuerySize = 1  % Default query size (scalar)
    end

    methods
        function layer = attentionLayer(attentionSize, hiddenStateDim, args)
            arguments
                attentionSize
                hiddenStateDim  % Must match BiLSTM output dimension
                args.Name = 'attention'
                args.QuerySize = 1
            end
            
            layer.Name = args.Name;
            layer.AttentionSize = attentionSize;
            layer.QuerySize = args.QuerySize;

            % Initialize parameters with correct dimensions
            layer.W_h = randn(attentionSize, hiddenStateDim);
            layer.W_s = randn(attentionSize, layer.QuerySize);
            layer.v = randn(1, attentionSize);
            layer.b = randn(attentionSize, 1);
            layer.s = randn(layer.QuerySize, 1);  % Learnable query vector
        end
    
        function Z = predict(layer, X)
                % X dimensions: [HiddenStateDim, SequenceLength, BatchSize] (dlarray)
                [hiddenStateDim, sequenceLength, batchSize] = size(X);
                
                % Reshape X for batch operations
                X_reshaped = reshape(X, hiddenStateDim, sequenceLength * batchSize);
                
                % Compute W_h * h_t
                Wh_ht = pagemtimes(layer.W_h, X_reshaped);
                Wh_ht = reshape(Wh_ht, layer.AttentionSize, sequenceLength, batchSize);
                
                % Compute W_s * s + b
                Ws_s = layer.W_s * layer.s;
                alignment_scores = tanh(Wh_ht + Ws_s + layer.b);
                
                % Compute scores: v * alignment_scores
                scores = pagemtimes(layer.v, alignment_scores); % [1, SequenceLength, BatchSize]
                
                % Manual softmax (with numerical stability)
                max_scores = max(scores, [], 2);
                exp_scores = exp(scores - max_scores);
                alpha = exp_scores ./ sum(exp_scores, 2); % [1, SequenceLength, BatchSize]
                
                % Reshape alpha for pagemtimes
                alpha_reshaped = permute(alpha, [2, 1, 3]); % [SequenceLength, 1, BatchSize]
                
                % Compute weighted sum
                Z = pagemtimes(X, alpha_reshaped); % [HiddenStateDim, 1, BatchSize]
            end
        end
    end
