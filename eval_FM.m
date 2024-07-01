function [val] = eval_FM(x, w, P)
    k = size(P, 1);
    val = dot(w, x);

    for f = 1:k
        val = val + 0.5 .* (dot(P(f, :), x).^2 - dot(P(f, :) .* P(f, :), x .* x));
    end

end
