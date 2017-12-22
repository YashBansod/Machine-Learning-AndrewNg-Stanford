function [J grad] = nnCostFunction(nn_params, input_layer_size, ...
                                    hidden_layer_size, num_labels, X, y, lambda)
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

% -------------------------------------------------------------

% Convert the y vector to a suitable form. Each row of Y should contain 
% num_labels columns each representing value from corresponding node of the
% output layer units.
Y = zeros(m, num_labels);
for i = 1: num_labels
    Y(y == i,i) = 1;
end

% Add the bias unit to the input
X = [ones(m , 1) X];

z2 = (Theta1 * X')';                        % Input to the Hidden layer
a2 = [ones(m, 1) sigmoid(z2)];              % Activation values of Hidden layer

z3 = (Theta2 * a2')';                       % Input to the output layer
H = sigmoid(z3);                            % Values at the output

% The first Row of Theta corresponds to the weight for the Bias Unit which
% shoud be equal to Zero while regularizing.
regTheta1 = Theta1.* [zeros(hidden_layer_size, 1), ones(size(Theta1, 1), size(Theta1, 2)-1)];
regTheta2 = Theta2.* [zeros(num_labels, 1), ones(size(Theta2, 1), size(Theta2, 2)-1)];
% Find the Cost due to regularization of parameters
regParaCost = (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2))) * (lambda / (2 * m));

% Caluculate the Cost
J = sum(sum(((-Y .* log(H)) - ((1-Y) .* log(1-H))) / m)) + regParaCost;

% Steps to caluculate the Gradient
% 1) Caluculate the del values which represent the "error" of each unit
del_3 = H - Y;
del_2 = (del_3 * Theta2) .* [zeros(m,1), sigmoidGradient(z2)];
del_2 = del_2(:, 2:end);

% 2) Caluculate the Delta values which correspond to the accumulated
% gradient from each layer
Delta_2 = zeros(num_labels, (hidden_layer_size + 1));
Delta_1 = zeros(hidden_layer_size, (input_layer_size + 1));
Delta_2 = Delta_2 + del_3' * a2;
Delta_1 = Delta_1 + del_2' * X;

% 3) Obtain the regularized gradient of the NN by taking the average of the
% accumulated gradient and adding the contribution due to regularization
% parameter applied on Theta.
Theta2_grad = (Delta_2 / m) + [zeros(num_labels, 1) Theta2(:, 2:end)]*(lambda/m);
Theta1_grad = (Delta_1 / m) + [zeros(hidden_layer_size, 1) Theta1(:, 2:end)]*(lambda/m);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
