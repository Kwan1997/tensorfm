function [w, P] = FM_CD2(X, y, k, beda_w, beda_P, verbose)
    [d, n] = size(X); % number of features and samples
    w = 1 .* randn(d, 1);
    P = 1 .* randn(k, d);
    y_pred = zeros(n, 1);

    for i = 1:n
        y_pred(i, 1) = eval_FM(X(:, i), w, P);
    end

    % lrate = realmax; % learning rate-free
    iter = 0;
    maxIter = 500;
    last_obj = realmax;
    obj = obj_FM2(X, y, w, P, n, d, beda_w, beda_P);

    sum_xipi = zeros(k, n);

    for i = 1:d
        sum_xipi = sum_xipi + repmat(P(:, i), [1, n]) * diag(X(i, :));
    end

    while (iter < maxIter)
        viol = 0;

        %% Update w
        for j = 1:d
            grad_wj = beda_w .* w(j, 1);
            eta_wj = beda_w;
            deriv_wj = zeros(n, 1);

            for i = 1:n
                deriv_wj(i, 1) = X(j, i);
                grad_wj = grad_wj + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) .* deriv_wj(i, 1);
                eta_wj = eta_wj + (1/n) * tfm_sqloss.mu * deriv_wj(i, 1)^2;
            end

            update = (1 / eta_wj) * grad_wj;
            w(j, 1) = w(j, 1) - update;
            viol = viol + abs(update);

            for i = 1:n
                y_pred(i, 1) = y_pred(i, 1) - update * deriv_wj(i, 1);
            end

        end

        %% Update P
        for j = 1:d

            sum_xipi = sum_xipi - repmat(P(:, j), [1, n]) * diag(X(j, :));

            for s = 1:k
                grad_P = beda_P .* P(s, j);
                eta_P = beda_P;
                deriv_P = zeros(n, 1);

                for i = 1:n
                    deriv_P(i, 1) = X(j, i) * sum_xipi(s, i);
                    grad_P = grad_P + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) * deriv_P(i, 1);
                    eta_P = eta_P + (1/n) * tfm_sqloss.mu * deriv_P(i, 1)^2;
                end

                update = (1 / eta_P) * grad_P;
                P(s, j) = P(s, j) - update;
                viol = viol + abs(update);

                for i = 1:n
                    y_pred(i, 1) = y_pred(i, 1) - update * deriv_P(i, 1);
                end

            end

            sum_xipi = sum_xipi + repmat(P(:, j), [1, n]) * diag(X(j, :));

        end

        obj = obj_FM2(X, y, w, P, n, d, beda_w, beda_P);

        if verbose
            fprintf("This is %d-th iteration, Obj is %f, viol is %f.\n", iter + 1, obj, viol);
        end

        if viol <= 1e-3
            break
        end

        iter = iter + 1;
        last_obj = obj;
    end

end
