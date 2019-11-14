function [Y, U, V] = RGBtoYUV(image)


T=[0.2126 0.7152 0.0722;
    -0.1146 -0.3854 0.5;
    0.5 -0.4542 -0.0458];

Inv_T = inv(T);

delta_1 = 16;
delta_2 = 128;
delta_3 = 128;


% file = 'kodim1.bmp';
% Ori_Img_rgb = double(imread(file));
Ori_Img_rgb = double(image);


%%%%%%%%%%%%%%%%%%%%%%  Load  RGB files %%%%%%%%%%%%%%%%%%%%%%%

R = Ori_Img_rgb(:,:,1);
G = Ori_Img_rgb(:,:,2);
B = Ori_Img_rgb(:,:,3);

[Row,Col]=size(R);

%================ RGB2YUV ==================%

YUV = T * [reshape(R,1,Row*Col);reshape(G,1,Row*Col);reshape(B,1,Row*Col)];
YUV = YUV+repmat([delta_1;delta_2;delta_3],1,size(YUV,2));

Y = reshape(YUV(1,:),Row,Col);
U = reshape(YUV(2,:),Row,Col);
V = reshape(YUV(3,:),Row,Col);

end