function im_out = tirbm_whiten_olshausen1_contrastnorm1(im)

if ~exist('D', 'var'), D = 16; end
if size(im,3)>1, im = rgb2gray(im); end
im = double(im);

im = im - mean(im(:));
im = im./std(im(:));

N1 = size(im, 1);
N2 = size(im, 2);

[fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
rho=sqrt(fx.*fx+fy.*fy)';

f_0=0.4*mean([N1,N2]);
filt=rho.*exp(-(rho/f_0).^4);

If=fft2(im);
imw=real(ifft2(If.*fftshift(filt)));

im_out = imw/std(imw(:)); % 0.1 is the same factor as in make-your-own-images

return
