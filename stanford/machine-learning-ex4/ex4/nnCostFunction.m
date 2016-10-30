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
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% one hot encoding
Y_one_hot = zeros( size( y, 1 ), num_labels );
for i = 1:num_labels     % assuming class labels start from one
    rows = y == i;
    Y_one_hot( rows, i ) = 1;
end


% forward propagation
A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);

A2 = [ones(size(A2,1), 1) A2];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);
H = A3;

J = 1 / m * sum(sum((-Y_one_hot .* log(H) - (1 - Y_one_hot) .* log(1 - H))));


% regularization
Theta1_temp = Theta1;
Theta1_temp(:,1) = 0;
Theta2_temp = Theta2;
Theta2_temp(:,1) = 0;

J_reg = lambda / (2*m) * (sum(sum(Theta1_temp.^2)) + sum(sum(Theta2_temp.^2)));
J = J + J_reg;





% grad_reg = lambda / m * theta;
% grad_reg(1) = 0;
% grad = 1 / m * X' * (h - y) + grad_reg;

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
Theta1_no_bias = Theta1(:,2:end);
Theta2_no_bias = Theta2(:,2:end);

for t = 1:m
  a1 = [1 X(t,:)];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);

  a2 = [1 a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  d3 = a3 - Y_one_hot(t,:);
  D2 = D2 + d3' * a2;

  d2 = (Theta2_no_bias' * d3')' .* sigmoidGradient(z2);
  D1 = D1 + d2' * a1;
end

Theta1_zero_bias = [zeros(size(Theta1_no_bias,1), 1) Theta1_no_bias];
Theta2_zero_bias = [zeros(size(Theta2_no_bias,1), 1) Theta2_no_bias];
Theta1_grad = D1/m + lambda/m*Theta1_zero_bias;
Theta2_grad = D2/m + lambda/m*Theta2_zero_bias;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end