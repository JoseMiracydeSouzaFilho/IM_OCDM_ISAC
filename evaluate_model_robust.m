function [accuracy, confMat] = evaluate_model_robust(net, X_val, Y_val)
    classes = categories(Y_val);
    num_classes = length(classes);
    try
        Y_pred_raw = predict(net, X_val);
        [~, Y_pred] = max(Y_pred_raw, [], 1);
        Y_pred = categorical(Y_pred', 1:num_classes, classes);
        
        % Enhanced evaluation metrics
        accuracy = mean(Y_pred == Y_val);
        confMat = confusionmat(Y_val, Y_pred);
        
        % Per-class metrics
        fprintf('\n=== Enhanced Evaluation Metrics ===\n');
        fprintf('Overall Accuracy: %.2f%%\n', accuracy*100);
        
        % Precision, Recall, F1 for each class
        for c = 1:num_classes
            TP = sum(Y_pred == classes(c) & Y_val == classes(c));
            FP = sum(Y_pred == classes(c) & Y_val ~= classes(c));
            FN = sum(Y_pred ~= classes(c) & Y_val == classes(c));
            
            precision = TP / (TP + FP + eps);
            recall = TP / (TP + FN + eps);
            f1 = 2 * (precision * recall) / (precision + recall + eps);
            
            fprintf('Class %d: Precision=%.2f, Recall=%.2f, F1=%.2f\n', ...
                c, precision, recall, f1);
        end
        
        % Confusion matrix with percentages
        confMatPercent = bsxfun(@rdivide, confMat, sum(confMat,2)) * 100;
        disp('Confusion Matrix (%):');
        disp(confMatPercent);
        
    catch ME
        fprintf('\nEvaluation error: %s\n', ME.message);
        accuracy = 0;
        confMat = zeros(num_classes);
    end
end