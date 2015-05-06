function vis_layer2=display_network_layer2(W2,W1)
for k2=1:size(W2,3)
        vis_layer2(:,:,k2) = conv2_mult(W2(:,:,k2),W1, 'full');
end
vis_layer=reshape(vis_layer2,size(vis_layer2,1)*size(vis_layer2,1),1,size(W2,3));
vis_layer2=vis_layer;
  display_network(vis_layer2)
end
function y = conv2_mult(B, a, convopt)
y = zeros(sqrt(size(B,1))+sqrt(size(a,1))-1,sqrt(size(B,1))+sqrt(size(a,1))-1);

for ii=1:size(a,3)
    b1=reshape(squeeze(B(:,ii)),sqrt(size(B,1)),sqrt(size(B,1)));
    a1=reshape(squeeze(a(:,:,ii)),sqrt(size(a,1)),sqrt(size(a,1)));
    y = y + conv2(b1, a1, convopt);
end
end
