function [x_train, y_train, x_test, y_test] = loadMNISTData(nTrainPerClass);
%loadMNISTData    Loads subset of MNIST handwritten digit data 
%
% Randomly shuffles the order of train and test sets,
%  so that all 1s aren't followed by all 2s, etc.
% This makes subsequent tasks (like building cross-validation datasets) interesting.
% INPUT
%   Ntrain : number of training examples to retain.
% OUTPUT
%   x_train : Ntrain x D matrix of training data
%   y_train : Ntrain x 1 vector of class labels (1, 2, or 3)
%   x_test : Ntest x D matrix of test data
%   y_test : Ntest x 1 vector of class labels (1,2, or 3)


% Load training and test data
Data = load('mnist_all.mat');

% Construct training set
x_train = [Data.train1(1:nTrainPerClass,:);
           Data.train2(1:nTrainPerClass,:);
           Data.train7(1:nTrainPerClass,:);
          ];

% class "3" denotes the number 7
y_train = [ones(nTrainPerClass, 1); 
           2*ones(nTrainPerClass, 1);
           3*ones(nTrainPerClass, 1);
          ]; 

% Construct corresponding test set
x_test = [Data.test1;
          Data.test2;
          Data.test7;
         ];
y_test = [ones(size(Data.test1,1), 1);
          2*ones(size(Data.test2,1),1);
          3*ones(size(Data.test7,1),1);
         ];
     
% Randomly shuffle the order of examples, 
%  so that we don't have all 1s followed by all 2s, etc.
rng('default');
randIDs = randperm(length(y_train));
x_train = x_train(randIDs,:);
y_train = y_train(randIDs,:);

randIDs = randperm(length(y_test));
x_test = x_test(randIDs,:);
y_test = y_test(randIDs,:);
     
% Ensure data type is "double" so calculations work as expected.
x_train = double(x_train);
x_test = double(x_test);
     
