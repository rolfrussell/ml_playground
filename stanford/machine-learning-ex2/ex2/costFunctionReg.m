function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



h = sigmoid(X*theta);

theta_temp = theta;
theta_temp(1) = 0;
J_reg = lambda / (2*m) * sum(theta_temp.^2);
J = 1 / m * sum((-y .* log(h) - (1 - y) .* log(1 - h))) + J_reg;

grad_reg = lambda / m * theta;
grad_reg(1) = 0;
grad = 1 / m * X' * (h - y) + grad_reg;



% =============================================================

end
