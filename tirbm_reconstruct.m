function negdata = tirbm_reconstruct(S, W, pars)

ws = sqrt(size(W,1));
patch_M = size(S,1);
patch_N = size(S,2);
numchannels = size(W,2);
numbases = size(W,3);

S2 = S;
negdata2 = zeros(patch_M+ws-1, patch_N+ws-1, numchannels);

for b = 1:numbases,
    H = reshape(W(:,:,b),[ws,ws,numchannels]);
    negdata2 = negdata2 + conv2_mult(S2(:,:,b), H, 'full');
end

negdata = 1*negdata2;

end
