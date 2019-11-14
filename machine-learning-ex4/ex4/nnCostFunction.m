function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Setup some useful variables
n = input_layer_size;
m = size(X, 1);
s2 = hidden_layer_size;
k = num_labels;

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % (s2 x n+1)
Theta2_grad = zeros(size(Theta2)); % (k x s2+1)

% Remove the bias units (the ones) for regularization terms
% Theta1 (s2 x n+1)
% Theta2 (k x s2+1)
T1 = [zeros(s2,1) Theta1(:,2:n+1)];
T2 = [zeros(k,1) Theta2(:,2:s2+1)];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
for i = 1:m
    % X                                (m x n)
    a1 = [1 X(i,:)];                 % (1 x n+1)
    z2 = a1 * Theta1';               % (1 x n+1) * (n+1 x s2) = (1 x s2)
    a2 = [1 sigmoid(z2)];            % (1 x s2+1)
    z3 = a2 * Theta2';               % (1 x s2+1) * (s2+1 x k) = (1 x k)
    a3 = sigmoid(z3);                % (1 x k)
    h = a3;                          % (1 x k)
    % y                                (m x 1)
    y_i = y(i,:);                    % (1 x 1)
    % Transform the label value to a vector where the indices are the labels
    % and only this label's index has a value of 1 and the other elements
    % are 0's
    y_k = zeros(1, k);               % (1 x k)
    y_k(1, y_i) = 1; 
    J = J + y_k*log(h') + (1-y_k)*log(1-h');
end

J = (-1/m) * J;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
D1 = Theta1_grad;
D2 = Theta2_grad;

for i = 1:m
    % Step 1 : propagate forward
    % X                                (m x n)
    a1 = [1 X(i,:)];                 % (1 x n+1)
    z2 = a1 * Theta1';               % (1 x n+1) * (n+1 x s2) = (1 x s2)
    a2 = [1 sigmoid(z2)];            % (1 x s2+1)
    z3 = a2 * Theta2';               % (1 x s2+1) * (s2+1 x k) = (1 x k)
    a3 = sigmoid(z3);                % (1 x k)
    
    % Step 2 : calculate this example's deltas for each layer
    % y                                (m x 1)
    y_i = y(i,:);                    % (1 x 1)
    % Transform the label value to a vector where the indices are the labels
    % and only this label's index has a value of 1 and the other elements
    % are 0's
    y_k = zeros(1, k);                                % (1 x k)
    y_k(1, y_i) = 1;
    d3 = a3 - y_k;                                    % (1 x k)
    % d3 * Theta2                                       (1 x k) * (k x s2+1) = (1 x s2+1)
    d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2); % (1 x s2) .* (1 x s2) = (1 x s2)
    
    % Step 3 : accumulate this example's deltas
    D1 = D1 + d2' * a1;      % D1 + (s2 x 1) * (1 x n+1) = (s2 x n+1)
    D2 = D2 + d3' * a2;      % D2 + (k x 1) * (1 x s2+1) = (k x s2+1)
end

% Theta1_grad (s2 x n+1)
% Theta2_grad (k x s2+1)
Theta1_grad = D1/m + (lambda/m) .* T1;
Theta2_grad = D2/m + (lambda/m) .* T2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

J = J + (lambda/(2*m)) * (sum(sum(T1.^2)) + sum(sum(T2.^2)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
