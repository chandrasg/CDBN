function [ferr dW_total dh_total dv_total poshidprobs poshidstates negdata] = ...
    fobj_tirbm_CD_LB_sparse(imdata, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg)

ws = sqrt(size(W,1)); %weight shape

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do convolution/ get poshidprobs
poshidexp = tirbm_inference(imdata, W, hbias_vec, pars); % For calculating bottom-up signal to hidden units from visible units
% poshidstates2 = double(poshidprobs > rand(size(poshidprobs))); 
[poshidstates poshidprobs] = tirbm_sample_multrand2(poshidexp, spacing); %sample hidden units (Eq 14)

posprods = tirbm_vishidprod_fixconv(imdata, poshidprobs, ws); 
poshidact = squeeze(sum(sum(poshidprobs,1),2));

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neghidstates = poshidstates;
for j=1:pars.K_CD  %% pars.K_CD-step contrastive divergence
    negdata = tirbm_reconstruct(neghidstates, W, pars); %Reconstruct visible units
    % neghidprobs = tirbm_inference(negdata, W, hbias_vec, pars);
    % neghidstates = neghidprobs > rand(size(neghidprobs));
    neghidexp = tirbm_inference(negdata, W, hbias_vec, pars); %Final step of CD - calculating hidden units again
    [neghidstates neghidprobs] = tirbm_sample_multrand2(neghidexp, spacing);
    
end
negprods = tirbm_vishidprod_fixconv(negdata, neghidprobs, ws); %Calculate the Negdata given hidden states (Sampling V from H)
neghidact = squeeze(sum(sum(neghidprobs,1),2));

ferr = mean( (imdata(:)-negdata(:)).^2 ); %Calculate the Recon error

if 0
    figure(1), display_images(imdata)
    figure(2), display_images(negdata)

    figure(3), display_images(W)
    figure(4), display_images(posprods)
    figure(5), display_images(negprods)

    figure(6), display_images(poshidstates)
    figure(7), display_images(neghidstates)
end


%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(bias_mode, 'none')
    dhbias = 0;
    dvbias = 0;
    dW = 0;
elseif strcmp(bias_mode, 'simple')
    dhbias = squeeze(mean(mean(poshidprobs,1),2)) - pars.pbias;
    dvbias = 0;
    dW = 0;
elseif strcmp(bias_mode, 'hgrad')
    error('hgrad not yet implemented!');
elseif strcmp(bias_mode, 'Whgrad')
    error('Whgrad not yet implemented!');
else
    error('wrong adjust_bias mode!');
end

numcases1 = size(poshidprobs,1)*size(poshidprobs,2);
% dW_total = (posprods-negprods)/numcases - l2reg*W - weightcost_l1*sign(W) - pars.pbias_lambda*dW;
dW_total1 = (posprods-negprods)/numcases1;
dW_total2 = - l2reg*W;
dW_total3 = - pars.pbias_lambda*dW;
dW_total = dW_total1 + dW_total2 + dW_total3;

dh_total = (poshidact-neghidact)/numcases1 - pars.pbias_lambda*dhbias;

dv_total = 0; %dv_total';

fprintf('||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n', sqrt(sum(W(:).^2)), sqrt(sum(dW_total1(:).^2)), sqrt(sum(dW_total2(:).^2)), sqrt(sum(dW_total3(:).^2)));

end
