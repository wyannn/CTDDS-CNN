
function Bits = BitcountOfBlock_JPEG_Chrom(Block)


if nnz(Block) ~= 0;

    B = ZigZag(Block,'HV');

    [run,size]=SymbolFormation_JPEG(B);

    AC_Bits = AC_BitCount_JPEG_Chrom(run,size);

    DC_Bits = DC_Bitcount_JPEG_Chrom(B);

    Bits = AC_Bits + DC_Bits;

else

    Bits = 0;

end

    
   