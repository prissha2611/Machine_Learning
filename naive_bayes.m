%Prissha Krishna Moorthy, 1001354261
function output = naive_bayes(training_file, test_file)

% training
data = load(training_file);
j = 1;
class = unique(data(:,end)); %classes list
for i=1:length(class)
    index = find(data(:,end)==class(i)); % rows of that class
    data1 = data(index,(1:end-1)); 
    data1_mean = mean(data1);
    data1_std = std(data1);
    for z =1:length(data1_std)
        if(data1_std(z) < 0.01)
            data1_std(z) = 0.01;
        end
    end
    
    for k=1:length(data1_mean)
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

data2 = load(test_file);

%p(x|C)
for i=1:size(data2,1) %going through each row
    for z=1:length(class) %going through each class
        m = 1;
        index = find(matrix(:,1)==z);
        for j = 1:length(index)
            x = data2(i,j);
            mu = matrix(index(j),3);
            s = matrix(index(j),4);
            gaussian = (1/(s*sqrt(2*pi)))*exp(-((x-mu).^2)/(2*(s.^2)));
            m = m*gaussian;
        end
        pxnc(i,z) = m;
    end
end

%p(x)
for i=1:size(pxnc,1)
    for j=1:size(pxnc,2)
        probX = pxnc * probC';
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
    [probability,predicted] = max(pcnx,[],2);
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







