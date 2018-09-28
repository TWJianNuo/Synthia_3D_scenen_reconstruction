img = imread('ngc6543a.jpg'); img = img(200:370,250:400,:); img = rgb2gray(img);
% figure(1); clf; imshow(img);
x = 2 : 0.01 : 3; y = ones(1, length(x)) * 5; interVal = zeros(length(x), 1);
for i = 1 : length(x)
    x1 = x(i); y1 = y(i);
    interVal(i) = interpImg(img,[y1 x1], false);
end
stem(interVal)
function interVal=interpImg(img,yx,zpad)
    if nargin<4,RGB=ndims(img);RGB(RGB<3)=1; end
    if nargin<3,zpad=true; end
    yx0=floor(yx);
    wt=yx-yx0; wtConj=1-wt;
    interTop=wtConj(2)*pixLookup(img,yx0(1),yx0(2),zpad,RGB)+wt(2)*pixLookup(img,yx0(1),yx(2),zpad,RGB);
    interBtm=wtConj(2)*pixLookup(img,yx(1),yx0(2),zpad,RGB)+wt(2)*pixLookup(img,yx(1),yx(2),zpad,RGB);
    interVal=wtConj(1)*interTop+wt(1)*interBtm;
end
function pixVal=pixLookup(img,y,x,zpad,RGB)
    % This helper function looks up a pixel value from a given input image
    % img is the input image (RGB or Grayscale)
    % yx is the coordinate and repEdge tells the condition for pixel values out
    % side of img (Use round up convention)
    % For grayscale use RGB =1
    if nargin<4,RGB=3;end
    
    pixVal=zeros(1,1,RGB); %Initialize the pixel
    
    if nargin<3
        zpad=true; %pad with black pixels
    end
    
    if RGB==3
        [ROW,COL,~]=size(img);
    else
        [ROW,COL]=size(img);
    end
    % If the pixel value is outside of image given
    if (x<=0)||(x>COL)||(y<=0)||(y>ROW)
        if zpad
            pixVal(:)=0;
        else
            y0=y;x0=x;
            y0(y0<1)=1; x0(x0<1)=1;
            y0(y0>ROW)=ROW;x0(x0>COL)=COL;
            % Bug fix suggested by Sanjuro
            pixVal=img(ceil(y0),ceil(x0),:);
        end
    else
        pixVal=img(ceil(y),ceil(x),:);
    end
end