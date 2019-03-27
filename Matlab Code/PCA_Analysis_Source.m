function a2
z1 = load('z1.dat');
z2 = load('z2.dat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATIONS FOR Z1%%%%%%%% Only code for z1 is commented since everything
%%%%%%%%%%%%%%%%%%%%%%%%%%%% done to z2 is the exact same.

%perform pcaprelim on z1
[sdiag, meanvec, uvecmat] = pcaprelim(z1);
%print the singular values
sdiag

%no scientific notation
format long g

%This is code to caculate the ratio of coverage given as sum of the first k
%eigen values over sum of all the eigen values, k was chosen so that the
%ratio is 0.50<k<0.60 :: p = tk/T
b = 1;
summer = 0;
for a = 1:10 
    for c = 1:b
        summer = summer + sdiag(c);
    end
    %unsupress the statement below to print the ratio of coverage for
    %increase number of eigen values 1-10
    d = summer/sum(sdiag);
    
summer = 0;
b = b + 1;
end


%first plot the eigen values
figure()
plot(sdiag)
title('Eigenvalues of z1')

%plot the mean vector returned by pca prelim 
figure()
%messy code to set labels and change scale to dates at bottom, done for
%most plots that have 100 x entries
t = datetime(1992,1,1) + calquarters(0:99);
plot(t,meanvec)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
title('Mean vector of z1')
xlabel('Date')
ylabel('Price')
approxnum = 4

%calculating the RMSE for every signal in z1 when reconstructed
e = zeros(1,18);
for c = 1:18
 new_data = z1(:,c);
 [approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
 RMSE = sqrt(mean((approxvec - new_data).^2));
 %store the rmse for every reconstructed signal into a new array
 e(c) = RMSE;
end

%plot the error trend for all signals
figure()
plot(e)
title('RMSE of reconstruction z1')
xlabel('Signal #')
ylabel('Error')
%norm error is the average error of reconstruction
normError = sum(e)/18

%PLOTTING INDIVIDUAL SIGNALS: Plot the original signal and the
%reconstructed signal in the same figure. Error is the difference between
%the lines.

%Plotting the most AVERAGE signal reconstructed (individual error closest
%to average)
new_data = z1(:,3);
[approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
figure()
t = datetime(1992,1,1) + calquarters(0:99);
%plot recontructed signal
plot(t,approxvec)
hold on
%plot original signal
plot(t,new_data)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
legend('Reconstructed Signal', 'Original Signal')
title('Most "Average" Signal Reconstruction in z1')
xlabel('Date')
ylabel('Price')

%Plotting the WORST signal reconstructed (highest RMSE)
new_data = z1(:,6);
[approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
figure()
t = datetime(1992,1,1) + calquarters(0:99);
plot(t,approxvec)
hold on
plot(t,new_data)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
legend('Reconstructed Signal', 'Original Signal')
title('"Worst" signal reconstruction z1')
xlabel('Date')
ylabel('Price')

%Plotting the BEST signal reconstructed (lowest error)
new_data = z1(:,7);
[approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
figure()
t = datetime(1992,1,1) + calquarters(0:99);
plot(t,approxvec)
hold on
plot(t,new_data)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
legend('Reconstructed Signal', 'Original Signal')
title('"Best" Signal Reconstructed in z1')
xlabel('Date')
ylabel('Price')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATIONS FOR Z2%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sdiag, meanvec, uvecmat] = pcaprelim(z2);
sdiag

format long g
b = 1;
summer = 0;
for a = 1:10
    for c = 1:b
        summer = summer + sdiag(c);
    end
    %unsupress the statement below to print the ratio of coverage for
    %increase number of eigen values 1-10
    d = summer/sum(sdiag);
    
summer = 0;
b = b + 1;
end

figure()
plot(sdiag)
title('Eigenvalues of z2')
figure()
t = datetime(1992,1,1) + calquarters(0:99);
plot(t,meanvec)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
title('Mean vector of z2')
xlabel('Date')
ylabel('Price')
approxnum = 4

e = zeros(1,18);

for c = 1:18
 new_data = z2(:,c);
 [approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
 RMSE = sqrt(mean((approxvec - new_data).^2));
 e(c) = RMSE;
end
figure()
plot(e)
title('RMSE of Reconstruction for z2')
xlabel('Signal #')
ylabel('Error')
normError = sum(e)/18

new_data = z2(:,18);
[approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
figure()
t = datetime(1992,1,1) + calquarters(0:99);
plot(t,approxvec)
hold on
plot(t,new_data)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
legend('Reconstructed Signal', 'Original Signal')
title('Most "Average" Signal Reconstruction in z2')
xlabel('Date')
ylabel('Price')

new_data = z2(:,17);
[approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
figure()
t = datetime(1992,1,1) + calquarters(0:99);
plot(t,approxvec)
hold on
plot(t,new_data)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
legend('Reconstructed Signal', 'Original Signal')
title('"Worst" signal reconstruction z2')
xlabel('Date')
ylabel('Price')

new_data = z2(:,12);
[approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec,uvecmat);
figure()
t = datetime(1992,1,1) + calquarters(0:99);
plot(t,approxvec)
hold on
plot(t,new_data)
xtickformat('MM-yyyy')
set(gca, 'XTick', (datetime(1992,1,1) : 365 : datetime(2016,1,12)));
legend('Reconstructed Signal', 'Original Signal')
title('"Best" Signal Reconstructed in z2')
xlabel('Date')
ylabel('Price')
end



function [sdiag, meanvec, uvecmat] = pcaprelim(Z)
% FUNCTION [SDIAG, MEANVEC, UVECMAT] = PCAPRELIM(Z)
% performs the preliminary Principal Components Analysis
% (PCA) of Z, a matrix in which the data are
% represented as columns. PCAPRELIM returns:
% SDIAG - singular values of the PCA, in decreasing order
% MEANVEC - the mean vector of the initial data
% UVECMAT - left singular vectors of the PCA, as column vectors
% Find the mean vector and form it into a matrix
[m,n] = size(Z);
meanvec = mean(Z,2);
M = meanvec*ones(1,n);
% Find the difference matrix
D = Z - M;
% Find the left singular vectors as a matrix and
% the singular values as a vector
[uvecmat, Smat, Vvecs] = svd(D, 'econ');
sdiag = diag(Smat);
end

function [approxcomp,approxvec] = pcaapprox(new_data,approxnum,meanvec, uvecmat)
% [APPROXCOMP,APPROXVEC]=PCAAPPROX(NEW_DATA, APPROXNUM,
% MEANVEC, UVECMAT)
% approximates new data based on a Principal Components Analysis
% (PCA) of initial data. Inputs are:
% NEW_DATA - a signal to be approximated, as a column vector
% APPROXNUM - a scalar giving the order of the approximation
% MEANVEC - the PCA mean vector (from PCAPRELIM)
% UVECMAT - the singular vectors of the PCA (from PCAPRELIM)
%
% Return values are:
% APPROXCOMP - the components as a row vector of scalars
% APPROXVEC - the approximation of the new data as a vector
% Set up the initial and return values
diffvec = new_data - meanvec;
approxcomp = zeros(approxnum, 1);
approxvec = meanvec;
% Loop through the eigenvectors, finding the components
% and building the approximation
for i=1:approxnum
uvec = uvecmat(:,i);
beta = dot(diffvec, uvec);
approxcomp(i,1) = beta;
approxvec = approxvec + beta*uvec;
end
end



