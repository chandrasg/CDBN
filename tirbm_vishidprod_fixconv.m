function vishidprod2 = tirbm_vishidprod_fixconv(imdata, H, ws)

numchannels = size(imdata,3);
numbases = size(H,3);

% tic
% TODO: single channel version is not implemented yet.. Might need to
% modify mexglx file
selidx1 = size(H,1):-1:1;
selidx2 = size(H,2):-1:1;
vishidprod2 = zeros(ws,ws,numchannels,numbases);

if numchannels==1
    vishidprod2 = conv2_mult(imdata, H(selidx1, selidx2, :), 'valid');
else
    for b=1:numbases
    for c=1:numchannels
         vishidprod2(:,:,c,b) = vishidprod2(:,:,c,b) + conv2_mult(imdata(:,:,c), H(selidx1, selidx2, b), 'valid');
     end
    end
end

vishidprod2 = reshape(vishidprod2, [ws^2, numchannels, numbases]);

end
