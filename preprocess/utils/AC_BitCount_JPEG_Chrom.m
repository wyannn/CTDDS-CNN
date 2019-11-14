%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  For AC Coefficients   %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% chrominance %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function NumberOfBits=AC_BitCount_JPEG_Chrom(run,size)

Count=[       2  3  4  5  5  6  7  9 10 12    4  6  8  9 11 12 16 16 16 16    5  8 10 12 15 16 16 16 16 16]; 
Count=[Count  5  8 10 12 16 16 16 16 16 16    6  9 16 16 16 16 16 16 16 16    6 10 16 16 16 16 16 16 16 16];
Count=[Count  7 11 16 16 16 16 16 16 16 16    7 11 16 16 16 16 16 16 16 16    8 16 16 16 16 16 16 16 16 16]; 
Count=[Count  9 16 16 16 16 16 16 16 16 16    9 16 16 16 16 16 16 16 16 16    9 16 16 16 16 16 16 16 16 16]; 
Count=[Count  9 16 16 16 16 16 16 16 16 16   11 16 16 16 16 16 16 16 16 16   14 16 16 16 16 16 16 16 16 16];
Count=[Count 10 15 16 16 16 16 16 16 16 16];

NumberOfBits=0;
K=length(run);

if K~=1

for j=2:K
    while run(j)>=15
        run(j)=run(j)-15;
        NumberOfBits=NumberOfBits+11;%% 11 bits for ZRL
    end
    Index=10*run(j)+size(j);
    if Index>0
        NumberOfBits=NumberOfBits+Count(Index);
    end
end

NumberOfBits=NumberOfBits+4; % 4 bits for EOB

else
    
    
   NumberOfBits=NumberOfBits+4; % 4 bits for EOB
   
end
    