function [S P] = multrand2(P)
% P is 2-d matrix: 2nd dimension is # of choices

% sumP = row_sum(P); 
sumP = sum(P,2);
P = P./repmat(sumP, [1,size(P,2)]);

cumP = cumsum(P,2);
% rand(size(P));
unifrnd = rand(size(P,1),1);
temp = cumP > repmat(unifrnd,[1,size(P,2)]);
Sindx = diff(temp,1,2);
S = zeros(size(P));
S(:,1) = 1-sum(Sindx,2);
S(:,2:end) = Sindx;

end
