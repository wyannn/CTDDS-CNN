%%%%%%%%%%%%%% DC chrominance %%%%%%%%%%%


function DC_Bit_chrom=DC_Bitcount_JPEG_Chrom(B)


count = [2 2 2 3 4 5 6 7 8 9 10 11];

if B(1) ~= 0

size = floor(log2(abs(B(1))))+1;

DC_Bit_chrom = count(size+1);

else
    
DC_Bit_chrom = count(1); %%DC = 0

end