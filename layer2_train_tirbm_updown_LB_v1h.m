function layer2_train_tirbm_updown_LB_v1h(images_all, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size,W1)

if mod(ws,2)~=0, error('ws must be even number'); end

sigma_start = 0.2;
sigma_stop = 0.1;

CD_mode = 'exp';
bias_mode = 'simple';

% Etc parameters
K_CD = 1;

% Initialization
 W = [];
 vbias_vec = [];
 hbias_vec = [];
 pars = [];

C_sigm = 1;

% learning
num_trials = 1000;

numchannels = size(images_all{1},3);

% Initialize variables
if ~exist('pars', 'var') || isempty(pars)
    pars=[];
end

if ~isfield(pars, 'ws'), pars.ws = ws; end
if ~isfield(pars, 'num_bases'), pars.num_bases = num_bases; end
if ~isfield(pars, 'spacing'), pars.spacing = spacing; end

if ~isfield(pars, 'pbias'), pars.pbias = pbias; end
if ~isfield(pars, 'pbias_lb'), pars.pbias_lb = pbias_lb; end
if ~isfield(pars, 'pbias_lambda'), pars.pbias_lambda = pbias_lambda; end
if ~isfield(pars, 'bias_mode'), pars.bias_mode = bias_mode; end

if ~isfield(pars, 'std_gaussian'), pars.std_gaussian = sigma_start; end
if ~isfield(pars, 'sigma_start'), pars.sigma_start = sigma_start; end
if ~isfield(pars, 'sigma_stop'), pars.sigma_stop = sigma_stop; end

if ~isfield(pars, 'K_CD'), pars.K_CD = K_CD; end
if ~isfield(pars, 'CD_mode'), pars.CD_mode = CD_mode; end
if ~isfield(pars, 'C_sigm'), pars.C_sigm = C_sigm; end

if ~isfield(pars, 'num_trials'), pars.num_trials = num_trials; end
if ~isfield(pars, 'epsilon'), pars.epsilon = epsilon; end

disp(pars)
% persistent ConvolvedFeatures;

%% Initialize weight matrix, vbias_vec, hbias_vec (unless given)
if ~exist('W', 'var') || isempty(W)
    W = 0.01*randn(pars.ws^2, numchannels, pars.num_bases);
end

if ~exist('vbias_vec', 'var') || isempty(vbias_vec)
    vbias_vec = zeros(numchannels,1);
end

if ~exist('hbias_vec', 'var') || isempty(hbias_vec)
    hbias_vec = -0.01*ones(pars.num_bases,1);
end


batch_ws = 28; % changed from 100 (2008/07/24)
imbatch_size = floor(100/batch_size);

fname_prefix = sprintf('../results/tirbm/layer2_tirbm_updown_LB_new1h_w%d_b%02d_p%g_pl%g_plambda%g_sp%d_CD_eps%g_l2reg%g_bs%02d_%s', ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size, datestr(now, 30));
fname_save = sprintf('%s', fname_prefix);
fname_mat  = sprintf('%s.mat', fname_save);
fname_out = fname_mat;
mkdir(fileparts(fname_save));
fname_out

initialmomentum  = 0.5;
finalmomentum    = 0.9;

error_history = [];
sparsity_history = [];

Winc=0;
vbiasinc=0;
hbiasinc=0;



for t=1:pars.num_trials
    % Take a random permutation of the samples
    tic;
    ferr_current_iter = [];
    sparsity_curr_iter = [];

    imidx_batch = randsample(length(images_all), imbatch_size, length(images_all)<imbatch_size);
    for i = 1:length(imidx_batch)
        imidx = imidx_batch(i);
        imdata = images_all{imidx};
        rows = size(imdata,1);
        cols = size(imdata,2);

        for batch=1:batch_size
            % Show progress in epoch
            fprintf(1,'epoch %d image %d batch %d\r',t, imidx, batch); 

               rowidx = ceil(rand*(rows-2*ws-batch_ws))+ws + [1:batch_ws];
               colidx = ceil(rand*(cols-2*ws-batch_ws))+ws + [1:batch_ws];
%  rowidx=[1:batch_ws];
%  colidx=[1:batch_ws];
            imdata_batch = imdata(rowidx, colidx,:);
            imdata_batch = imdata_batch - mean(imdata_batch(:));
            imdata_batch = trim_image_for_spacing_fixconv(imdata_batch, ws, spacing);
            
            if rand()>0.5,
%                 imdata_batch = fliplr(imdata_batch(:,:,1:24));
            end
            
            % update rbm
            [ferr dW dh dv poshidprobs poshidstates negdata]= fobj_tirbm_CD_LB_sparse(imdata_batch, W, hbias_vec, vbias_vec, pars, CD_mode, bias_mode, spacing, l2reg);
%             ConvolvedFeatures=cat(4,ConvolvedFeatures,poshidstates);
            ferr_current_iter = [ferr_current_iter, ferr];
            sparsity_curr_iter = [sparsity_curr_iter, mean(poshidprobs(:))];

            if t<5,
                momentum = initialmomentum;
            else
                momentum = finalmomentum;
            end

            % update parameters
            Winc = momentum*Winc + epsilon*dW;
            W = W + Winc;

            vbiasinc = momentum*vbiasinc + epsilon*dv;
            vbias_vec = vbias_vec + vbiasinc;

            hbiasinc = momentum*hbiasinc + epsilon*dh;
            hbias_vec = hbias_vec + hbiasinc;
        end
        mean_err = mean(ferr_current_iter);
        mean_sparsity = mean(sparsity_curr_iter);

        if (pars.std_gaussian > pars.sigma_stop) % stop decaying after some point
            pars.std_gaussian = pars.std_gaussian*0.99;
        end

        % figure(1), display_network(W);
        % figure(2), subplot(1,2,1), imagesc(imdata(rowidx, colidx)), colormap gray
        % subplot(1,2,2), imagesc(negdata), colormap gray
    end
    toc;

    error_history(t) = mean(ferr_current_iter);
    sparsity_history(t) = mean(sparsity_curr_iter);

figure(1), display_network_layer2(W,W1);
     if mod(t,10)==0,
        saveas(gcf, sprintf('%s_%04d.png', fname_save, t));
     end

%     fprintf('epoch %d error = %g \tsparsity_hid = %g\n', t, mean(ferr_current_iter), mean(sparsity_curr_iter));
%      save('layer2.mat', 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');

    disp(sprintf('results saved as %s\n', fname_mat));
  
    if mod(t, 10) ==0
        fname_timestamp_save = sprintf('%s_%04d.mat', fname_prefix, t);
        save(fname_timestamp_save, 'W', 'pars', 't', 'vbias_vec', 'hbias_vec', 'error_history', 'sparsity_history');
    end

end
save layer2.mat 
end
