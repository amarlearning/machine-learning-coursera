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

    % size of X is 5000 x 400
    % size of y is 5000 x 1

    % size of Theta 1 is 25 x 401
    % size of Theta 2 is 10 x 26

    % PART 1 : FEED FORWARD AND COST FUCNTION REGULARISATION

    theta1NoBias = Theta1(:, 2:end);
    theta2NoBias = Theta2(:, 2:end);

    K = num_labels;
    Y = eye(K)(y, :);

    A1 = [ones(m, 1), X];

    Z2 = A1 * Theta1';
    A2 = sigmoid(Z2);

    A2 = [ones(size(A2, 1), 1), A2];

    Z3 = A2 * Theta2';
    A3 = sigmoid(Z3);

    cost = sum(((-Y .* log(A3)) - ((1 - Y) .* log((1 - A3)))), 2);

    J = ((1 / m) .* sum(cost)) + ((lambda / (2 * m)) * (sum(sumsq(theta1NoBias)) + sum(sumsq(theta2NoBias))));

    % -------------------------------------------------------------

    % PART 2 : BACKPROPOGATION ALGORITHM

    Delta1 = 0;
    Delta2 = 0;

    for r = 1:m

        % adding bias unit in input unit : A1 = 401 * 1
        A1 = [1; X(r, :)'];

        Z2 = Theta1 * A1;

        % adding bias to the A2 : 26 * 1
        A2 = [1; sigmoid(Z2)];

        Z3 = Theta2 * A2;
        A3 = sigmoid(Z3);

        % here Y is 5000 * 10, each row has output value of input image
        D3 = A3 - Y(r, :)';

        % z2 is 25 * 1
        D2 = (theta2NoBias' * D3) .* sigmoidGradient(Z2);

        Delta2 += (D3 * A2');
        Delta1 += (D2 * A1');

    end

    Theta1_grad = (1 / m) * Delta1;
    Theta2_grad = (1 / m) * Delta2;

    % -------------------------------------------------------------

    % PART 3 : Regularized Gradient

    Theta1_grad(:, 2:end) += ((lambda / m) * theta1NoBias);
    Theta2_grad(:, 2:end) += ((lambda / m) * theta2NoBias);

    % -------------------------------------------------------------

    % =========================================================================

    % Unroll gradients
    grad = [Theta1_grad(:); Theta2_grad(:)];

end
