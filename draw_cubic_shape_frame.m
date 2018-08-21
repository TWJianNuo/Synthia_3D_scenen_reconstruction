function draw_cubic_shape_frame(cuboid)
    col = 'r';
    pts1 = cuboid{1}.pts;
    pts2 = cuboid{3}.pts;
    cornerpoints = [pts1; pts2];
    line_width = 2;
    hx = cornerpoints(:,1);hy = cornerpoints(:,2);hz = cornerpoints(:,3);
    
    x=[hx(1);hx(2)];y=[hy(1);hy(2)];z=[hz(1);hz(2)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(2);hx(3)];y=[hy(2);hy(3)];z=[hz(2);hz(3)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(3);hx(4)];y=[hy(3);hy(4)];z=[hz(3);hz(4)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(4);hx(1)];y=[hy(4);hy(1)];z=[hz(4);hz(1)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(5);hx(6)];y=[hy(5);hy(6)];z=[hz(5);hz(6)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(6);hx(7)];y=[hy(6);hy(7)];z=[hz(6);hz(7)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(7);hx(8)];y=[hy(7);hy(8)];z=[hz(7);hz(8)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(8);hx(5)];y=[hy(8);hy(5)];z=[hz(8);hz(5)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(1);hx(6)];y=[hy(1);hy(6)];z=[hz(1);hz(6)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(2);hx(5)];y=[hy(2);hy(5)];z=[hz(2);hz(5)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(3);hx(8)];y=[hy(3);hy(8)];z=[hz(3);hz(8)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(4);hx(7)];y=[hy(4);hy(7)];z=[hz(4);hz(7)];plot3(x,y,z,col,'LineWidth',line_width);hold off;
end