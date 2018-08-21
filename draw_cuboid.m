function draw_cuboid(cuboid)
    for i = 1 : 5
        hold on;
        patch('Faces',[1 2 3 4],'Vertices',cuboid{i}.pts,'FaceAlpha', 0.5)
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
    grid on
    axis equal
end