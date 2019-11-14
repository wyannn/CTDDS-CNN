function rgb = YUVtoRGB(image)


T=[0.2126 0.7152 0.0722;
    -0.1146 -0.3854 0.5;
    0.5 -0.4542 -0.0458];

Inv_T = inv(T);

delta_1 = 16;
delta_2 = 128;
delta_3 = 128;


% file = 'kodim1.bmp';
% Ori_Img_rgb = double(imread(file));
Ori_Img_yuv = image;


%%%%%%%%%%%%%%%%%%%%%%  Load  RGB files %%%%%%%%%%%%%%%%%%%%%%%

Y = Ori_Img_yuv(:,:,1);
U = Ori_Img_yuv(:,:,2);
V = Ori_Img_yuv(:,:,3);

[Row,Col]=size(Y);

%================ YUVtoRGB ==================%

Rec_YUV(1,:) = reshape(Y, 1, Row * Col);
Rec_YUV(2,:) = reshape(U, 1, Row * Col);
Rec_YUV(3,:) = reshape(V, 1, Row * Col);
Rec_YUV = Rec_YUV - repmat([delta_1;delta_2;delta_3], 1, Row * Col);
Rec_RGB = Inv_T * Rec_YUV;

R = reshape(Rec_RGB(1,:),Row,Col);
G = reshape(Rec_RGB(2,:),Row,Col);
B = reshape(Rec_RGB(3,:),Row,Col);

rgb(:,:,1) = R;
rgb(:,:,2) = G;
rgb(:,:,3) = B;


end