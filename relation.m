clc; clear all
mx = 50;
my = 50; 

x = [];

for i=0:mx-1
   for j = 0:my-1
       x = cat(1,x,[i,j,1]);
   end
end

y = [0:mx*my-1]';

w1 = x\y
x(:,3) =[];
y = cat(2,y,ones(length(y),1));
size(y);
w2 = y\x

