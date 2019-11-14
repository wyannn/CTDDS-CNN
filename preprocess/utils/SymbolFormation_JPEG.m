function [Run,Level]=SymbolFormation_JPEG(B)

Length=length(B);
I=1; 
J=1;

if sum(abs(B))~=0

while sum(abs(B))~=0
    while sum(abs(B(1:I)))==0
        I=I+1;
    end
    Run(J)=I-1;
    Level(J)=1+floor(log2(abs(B(I))));
    if I<length(B)
        B=B(I+1:length(B));
        J=J+1; 
        I=1;
    else
         break
        
    end    

end

else
    
    Run = 0;
    Level = 0;
    
end
    