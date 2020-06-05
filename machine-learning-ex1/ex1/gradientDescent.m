function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %sum = zeros(2, 1);
    %for i = 1:m
    %  tmp = theta(1) + theta(2) * X(i, 2) - y(i);
    %  sum(1) += tmp;
    %  sum(2) += tmp * X(i, 2);
    %endfor
    %theta(1) -= (alpha / m) * sum(1);
    %theta(2) -= (alpha / m) * sum(2);

    %i=1:m;
    %newTheta1 = theta(1) - alpha/m*sum(theta(1)+theta(2).*X(i,2)-y(i));
    %theta(2) -= alpha/m*sum((theta(1)+theta(2).*X(i,2)-y(i)).*X(i,2));
    %theta(1) = newTheta1;
    
    sqrErrors = X * theta - y;
    theta(1) -= alpha/m * sum(sqrErrors);
    theta(2) -= alpha/m * sum(sqrErrors .* X(:,2));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
