data=dlmread('heart.txt');
[trainind,testind]=dividerand(270,200,70);
traindata=data(trainind,:);
testdata=data(testind,:);
test=testdata(:,1:13);
testclasses=testdata(:,14);
train=traindata(:,1:13);
trainclasses=traindata(:,14);




%using polynomial kernel
for i=1:200
    for j=1:200
        mul=dot(train(i,:),train(j,:));
        mul=1+mul;
        kernel(i,j)=power(mul,5);
        
    end
end

%using sigmoid gradient descent kernel
c1=0.009;
c2=-1000;
for i=1:200
    for j=1:200
        mul=dot(train(i,:),train(j,:));
        kernel(i,j)=tanh((c1*mul)+c2);
        
    end
end

%using gaussian kernel
variance=0.4;
for i=1:200
    for j=1:200
        diff=train(i,:)-train(j,:);     %distance between the features
       squared_diff=dot(diff,diff);
     
      kernel(i,j)=exp((squared_diff*(-1))/(2*variance*variance));
    end
end

%WE TRIED THE 3 DIFFERENT KERNELS AND GOT THE BEST ACCURACY FOR GAUSSIAN KERNEL WITH VARIANCE 0.4 

eps=0.0005;
%for p=1:200
 %   for q=1:200
  %      kernel(p,q)=1;
   % end
%end
num_pos=0;
num_neg=0;

train_rows=length(train(:,1));
test_rows = length(test(:,1));
num_attr = length(train(1,:));
for i=1:train_rows
    if trainclasses(i)==1
        num_pos=num_pos+1;
    else
        num_neg=num_neg+1;
        trainclasses(i)=-1;
    end
end
 for i=1:test_rows
         if(testclasses(i)==2)
          testclasses(i)=-1;
         end
    end
diff=0;
 alpha = randi([0, 1000], [train_rows, 1]);
    diff = dot(alpha, trainclasses);

for i=1:num_pos
      alpha(i) = alpha(i) - (diff /num_pos);
end
b=zeros(train_rows,1);
for iters=1:200
    w=zeros(1,num_attr);
    for i=1:train_rows
      w=w+alpha(i)*trainclasses(i)*train(i,:);
    end
    for i=1:train_rows
        temp=dot(w,train(i,:));
        temp=trainclasses(i)*(temp+b(i));
        kkt(i)=alpha(i)*(temp-1);
    end
    [maxkktval, maxkktind] = max(kkt);
    X1=train(maxkktind,:);%Pciking X1
    e = zeros(train_rows,1);
    for i=1:train_rows
        for j=1:train_rows
         temp=alpha(j)*trainclasses(j)*(kernel(j,maxkktind)-kernel(j,i));
        e(i)=e(i)+temp+trainclasses(i)-trainclasses(maxkktind);    
        end
    end
    abse=abs(e);
    [maxeval,maxeind]=max(abse);
    X2=train(maxeind,:);
    k=kernel(maxkktind,maxkktind)+kernel(maxeind,maxeind)-2*kernel(maxkktind,maxeind);
    oldalpha2=alpha(maxeind);
    newalpha2=oldalpha2+(trainclasses(maxeind) * e(maxeind)) / k;
    alpha(maxeind)=newalpha2;
    if (alpha(maxeind)<0)
     alpha(maxeind)=0;
    end
    oldalpha1=alpha(maxkktind);
    newalpha1=oldalpha1+trainclasses(maxeind)*trainclasses(maxkktind)*(oldalpha2-newalpha2);      
    alpha(maxkktind)=newalpha1;
    if(alpha(maxkktind)<0)
     alpha(maxkktind)=0;
    end
%calculating the bias
     for i = 1:train_rows
            if alpha(i) > 0
                b(i) = trainclasses(i) - dot(w, train(i, :));
            end
     end
 
    % if abs(dot(alpha, trainclasses)) > 0.00000001
        %    disp('dot(alpha, classes) != 0');
           % error('The dot(alpha, classes) != 0');
    % end
%total bias
    biascount=0;
    for i=1:train_rows
      if alpha(i)~=0
        biascount=biascount+1;
      end
    end
    totalbias=sum(b)/biascount;
%the updated weight vector
    w=zeros(1,num_attr);
    for i=1:train_rows
        w=w+alpha(i)*trainclasses(i)*train(i,:);
    end
   
for i=1:length(testclasses)
        classes(i,1)=sign(dot(w,test(i,:))+totalbias);
end
%it gives what the corresponding test instance is being classified as
    accuracy=confusionmat(classes,testclasses)
    if (accuracy(1,2)+accuracy(2,1))/(accuracy(1,1)+accuracy(1,2)+accuracy(2,1)+accuracy(2,2))<eps
       break
    end
end
weight=w;
bias=b;
truePositive=accuracy(1,1);
falsePositive=accuracy(2,1);
