close all
x = pi/2;
y1_ =@(x) sin(x); y2_ =@(y) (4 - y).^2;
j1_ =@(x) cos(x); diff_ = @(y) 4 - y;
% y1_ =@(x) x; y2_ =@(y) (4 - y).^2;
% j1_ =@(x) 1; diff_ = @(y) 4 - y;

it_num = 1000;
val_record = zeros(it_num, 1); delta_record = zeros(it_num, 1); x_record = zeros(it_num, 1);
J_record = zeros(it_num, 1); y_record = zeros(it_num, 1);
for i = 1 : it_num
    y = y1_(x);
    y2 = y2_(y);
    diff = diff_(y);
    j1 = j1_(x);
    if abs(j1) < 0.01
        j1 = 0.01;
    end
    delta = j1 * diff / j1 / j1;
    x =  x + 0.001 * delta;
    val_record(i) = y2; delta_record(i) = delta; x_record(i) = x; J_record(i) = j1; y_record(i) = y;
end


figure(1); stem(val_record, 'Marker', '.')
figure(2); stem(delta_record, 'Marker', '.')
figure(3); stem(x_record, 'Marker', '.')
figure(4); stem(J_record, 'Marker', '.')
figure(5); stem(y_record, 'Marker', '.')