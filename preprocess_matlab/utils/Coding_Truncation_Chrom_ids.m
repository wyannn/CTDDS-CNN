
function [Rec_Img_Trun, bits]=Coding_Truncation_Chrom_ids(U,quality)

D_8x8 = dctmtx(8);
T_64x64 = kron(D_8x8',D_8x8');
T=[];
for i=1:4
    T = [T,T_64x64(:,(8*i-7):(8*i-4))];
end

Q_table = jpeg_qtable(quality, 1, 1);

table = Q_table(1:4, 1:4);
weights = maxmin(table, 1, 1.05);
weights = weights(:);
W = diag(weights);

Blk_size = 8;

[row,col] = size(U);
%===================  Initialization =====================%
Num_1 = row/Blk_size;
Num_2 = col/Blk_size;

Num_cell = Num_1*Num_2;

Ori_Block = mat2cell(U,Blk_size*ones(1,Num_1),Blk_size*ones(1,Num_2));

%================ Truncation-based Coding ==================%

Rec_Block = cell(Num_1,Num_2);

for k = 1:Num_cell
    
    Block = Ori_Block{k};
    coef_64x1 = SolveFISTA_2(T, Block(:), 50, 0);
    coef_64x1 = inv(W) * coef_64x1;
    DCT_Block = reshape(coef_64x1, [4, 4]);
    
    Pad_Block = [DCT_Block zeros(4,4); zeros(4,8)];
    
    Quan_Block = round(Pad_Block./Q_table);  %%%% Quantization
    Deq_Block = Quan_Block.*Q_table;    %%%% Dequantization
    
    Bits_Block_JPEG(k) = BitcountOfBlock_JPEG_Chrom(Quan_Block); 
    
    Rec_Block{k} = idct2(Deq_Block); %%%%2-D IDCT by scaling factor 2
    
    
end

Rec_Img_Trun = round(cell2mat(Rec_Block));
bits = sum(Bits_Block_JPEG) / row / col;

