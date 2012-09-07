% doublify_test.m

clear
clc

% add the folder containing the new function to your path
%addpath('/your/path/to/jacketSDK/doublify');

N = 10;
f = grand(N,1);

fprintf('f before...\n');
f

f_doubled = doublify(f);

fprintf('f after....\n');
f_doubled
