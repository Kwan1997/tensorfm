function [val] = obj_FM2(X, y, w, P, n, d, beda_w, beda_P)
    val = 0.5 .* (beda_w .* norm(w).^2 + beda_P * norm(P, 'fro').^2);

    for i = 1:n

        val = val + (1/n) * tfm_sqloss.loss(eval_FM(X(:, i), w, P), y{i});

    end
    
end
