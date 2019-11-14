function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta(:,1));
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% This is a vector of m elements esitmated by hypothesis
h = sigmoid(X * theta);

for i=1:m
    J = J - (1/m) * (y(i) * log(h(i)) + (1 - y(i)) * log(1 - h(i)));
end

% add the regularization term
J = J + (lambda/(2*m)) * [0, ones(1, n - 1)] * theta.^2;

%grad = grad + (1/m) * X' * (h - y) + (lambda/m) * [0, ones(1, n - 1)] * theta;
grad = grad + (1/m) * X' * (h - y);
for j=2:n
    grad(j) = grad(j) + (lambda/m) * theta(j);
end
% =============================================================

end
