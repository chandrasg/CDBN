function [poshidexp2] = tirbm_inference(imdata, W, hbias_vec, pars)

ws = sqrt(size(W,1));
numbases = size(W,3);
numchannel = size(W,2);

poshidprobs2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases);
poshidexp2 = zeros(size(imdata,1)-ws+1, size(imdata,2)-ws+1, numbases); 
for c=1:numchannel
    H = reshape(W(end:-1:1, c, :),[ws,ws,numbases]); % reshaping W as Wk (dividing into groups)
    poshidexp2 = poshidexp2 + conv2_mult(imdata(:,:,c), H, 'valid'); % part 2 of Equation 13 - (~Wk*v)ij
end

for b=1:numbases % b = small k in eq 13
    poshidexp2(:,:,b) = 1/(pars.std_gaussian^2).*(poshidexp2(:,:,b) + hbias_vec(b)); % Equation 13 - calculating bottom-up signal to hidden unit in every group
    poshidprobs2(:,:,b) = 1./(1 + exp(-poshidexp2(:,:,b))); %Eq 15 - Sampling p{a,k} - pooling unit of kth group in block a - No unit being active
end

end
