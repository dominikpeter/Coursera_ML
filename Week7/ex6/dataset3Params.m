function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_values = [1, 3, 10];
sigma_values =  [0.1, 0.3, 0.7];
error = inf;

counter = 1;
number_of_iters = numel(C_values) * numel(sigma_values);


for c = C_values
  for s = sigma_values    
    fprintf('Training Model %i of %i \n', counter, number_of_iters);
    fprintf('Training Model with Sigma = %f and C = %f \n', s, c);
    model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
    e = mean(double(svmPredict(model, Xval) ~= yval));
    if( e <= error )
      fprintf('Optimal Values updated with Sigma = %f and C = %f\n', ...
              s, c);
      C = c;
      sigma = s;
      error = e;
    endif
   counter += 1;
   endfor
 endfor



% =========================================================================

end
