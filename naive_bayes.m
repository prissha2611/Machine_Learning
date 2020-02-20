%Prissha Krishna Moorthy, 1001354261
function output = naive_bayes(training_file, test_file)

% training
data = load(training_file); %upload training file
j = 1;
class = unique(data(:,end)); %the different classes in the data
for i=1:length(class)
    index = find(data(:,end)==class(i)); %finds the row index of the class
    data1 = data(index,(1:end-1)); %obtain the data of the class
    data1_mean = mean(data1); %the mean of each attribute of the asme class
    data1_std = std(data1); %the std of each attribute of the same class
    for z =1:length(data1_std)
        if(data1_std(z) < 0.01)
            data1_std(z) = 0.01; % set 0.01 for std that is lower than 0.01
        end
    end
    
    for k=1:length(data1_mean) % Adding all class, attribute, mean and std
        matrix(j,1) = class(i);
        matrix(j,2) = k;
        matrix(j,3) = data1_mean(k);
        matrix(j,4) = data1_std(k);
        j=j+1;
    end
end

for i=1:size(matrix,1)
   fprintf('Class %d, attribute %d, mean = %.2f, std = %.2f\n',matrix(i,1), matrix(i,2), matrix(i,3), matrix(i,4));
end
fprintf('\n\n');

% p(C) probability of class
for i= 1:length(class)
    index = find(data(:,end)==i);
    probC(i) = length(index)/size(data,1);
end

data2 = load(test_file); %load test data

%p(x|C)
for i=1:size(data2,1) %going through each row of test dat
    for z=1:length(class) %going through each class
        pdf = 1;
        index = find(matrix(:,1)==z);
        for j = 1:length(index)
            x = data2(i,j);
            mu = matrix(index(j),3);
            s = matrix(index(j),4);
            gaussian = (1/(s*sqrt(2*pi)))*exp(-((x-mu).^2)/(2*(s.^2)));
            pdf = pdf*gaussian; %probability density function for each class
        end
        pxnc(i,z) = pdf; % each row contains the pdf of each class 
    end
end

%p(x)
for i=1:size(pxnc,1)
    for j=1:size(pxnc,2)
        probX = pxnc * probC'; %sum rule
    end
end

% bayes rule
for i=1:size(pxnc,1)
    for j=1:size(pxnc,2)
        pcnx(i,j) = (pxnc(i,j)*probC(j))/probX(i);
    end
end

%prediction
for i = 1:size(pcnx,1)
    [probability,predicted] = max(pcnx,[],2); % choosing the highest
end


%accuracy
for i = 1:size(pcnx,1)
    index = find(pcnx(i,:)==probability(i));
    if(length(index) > 1)
        accuracy(i) = 1/length(index);
    elseif(index == data2(i,end))
        accuracy(i) = 1;
    else
        accuracy(i) = 0;
    end
end


for i =1:size(pcnx,1)
    fprintf('ID= %5d, predicted= %3d, probability = %.4f, true= %3d, accuracy= %4.2f\n',i,predicted(i),probability(i),data2(i,end),accuracy(i));
end

fprintf('\nclassification accuracy= %6.4f\n',(sum(accuracy)/length(accuracy))*100);

end







