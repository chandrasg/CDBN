function [H HP] = tirbm_sample_multrand2(poshidexp, spacing)
% poshidexp is 3d array
poshidprobs = exp(poshidexp); %Shouldnt there be a denominator here?
poshidprobs_mult = zeros(spacing^2+1, size(poshidprobs,1)*size(poshidprobs,2)*size(poshidprobs,3)/spacing^2);
poshidprobs_mult(end,:) = 1;
% TODO: replace this with more realistic activation, bases..
for c=1:spacing
    for r=1:spacing
        temp = poshidprobs(r:spacing:end, c:spacing:end, :);
        poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
    end
end

[S1 P1] = multrand2(poshidprobs_mult');
S = S1';
P = P1';
clear S1 P1

% convert back to original sized matrix
H = zeros(size(poshidexp));
HP = zeros(size(poshidexp));
for c=1:spacing
    for r=1:spacing
        H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
        HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
    end
end


end
