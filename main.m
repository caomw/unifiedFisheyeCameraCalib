% This script is used to find the xi parameter for unified fisheye
% projection model, given the equivalent equidistant fisheye model.
clc; clear all;
%% camera parameters
width = 640;
height = 480;

polynomial_(1) = -179.471829787234; 
polynomial_(2) = 0; 
polynomial_(3) = 0.002316743975; 
polynomial_(4) = -3.635968439375e-06; 
polynomial_(5) = 2.0546506810625e-08;

cx = 320;
cy = 240;

distortion0 = 1.0;
distortion1 = 0.0;
distortion2 = 0.0;

affine_correction = [1.0, distortion2;...
                    distortion1, distortion0];
                
affine_correction_inverse = inv(affine_correction);     
%% back-projection using omidirectional model

MAX_PTS = 60000;

keypointsInPixel = zeros(2, MAX_PTS);
rays = zeros(3, MAX_PTS);

for i = 1:MAX_PTS
    u = randi(639);
    v = randi(479);
    
    while u == cx
       u = randi(639); 
    end
    
    while v == cy
       v = randi(439); 
    end
    
    keypointsInPixel(:, i) = [u; v];
    keypointInPixel = [u; v];
    
    principlePoint = [cx; cy];
    rectified = affine_correction_inverse*(keypointInPixel - principlePoint);

    rho = norm(rectified);

    ray = zeros(3, 1);
    ray(1:2, :) = rectified;

    ray(3) = polynomial_(5);
    ray(3) = polynomial_(4) + ray(3) * rho;
    ray(3) = polynomial_(3) + ray(3) * rho;
    ray(3) = polynomial_(2) + ray(3) * rho;
    ray(3) = polynomial_(1) + ray(3) * rho;
    ray(3) = (-1.0) * ray(3);

    rays(:, i) = ray/norm(ray);
end

%% solve for fx, fy, xi in least square form
% Ax = b, x = [fx, fy, xi]
A = zeros(2*MAX_PTS, 3);
b = zeros(2*MAX_PTS, 1);
scaleFactor = 30000; % to avoid numerical problem

for i = 1:MAX_PTS
   xs = rays(1, i);
   ys = rays(2, i);
   zs = rays(3, i);
   
   u = keypointsInPixel(1, i);
   v = keypointsInPixel(2, i);
   
   A(2*i - 1, 1) = xs; A(2*i - 1, 3) = (cx - u)/scaleFactor;
   A(2*i, 2) = ys;     A(2*i, 3) = (cy - v)/scaleFactor;
   
   b(2*i - 1) = (u - cx)*zs;
   b(2*i) = (v - cy)*zs;
end

%% minimize Ax - b
x = A\b;
fx = x(1);
fy = x(2);
xi = x(3)/scaleFactor;

X0 = [fx, fy, xi, cx, cy, 0, 0, 0, 0, 0]
% 
% data4ceres = zeros(MAX_PTS, 5);
% data4ceres(:, 1:2) = keypointsInPixel';
% data4ceres(:, 3:5) = rays';
% 
% save('data4ceres.dat', 'data4ceres', '-ascii');
% 
% % 
options = optimoptions(@lsqnonlin,'Algorithm','trust-region-reflective',...
    'MaxFunEvals',1500*5, ...
    'TolX', 1e-12, ...
    'TolFun', 1e-12);

obj_fcn = @(x) reprojection_error(x, keypointsInPixel, rays, MAX_PTS);

[x] = lsqnonlin(obj_fcn, X0, [], [], options);

%% do verifications
fx = x(1);
fy = x(2);
xi = x(3);
cx = x(4); 
cy = x(5);
k1 = x(6);
k2 = x(7);
k3 = x(8);
p1 = x(9);
p2 = x(10);

errors = zeros(MAX_PTS, 1);

for i = 1:10
    xs = rays(1, i); ys = rays(2, i); zs = rays(3, i);
    u = keypointsInPixel(1, i); v = keypointsInPixel(2, i);
    
   % project to normalized image plane
    mu = xs/(zs + xi);
    mv = ys/(zs + xi);
    
    % add radial distortion
    rho = sqrt(mu^2 + mv^2);
    distortion = 1 + k1*rho^2 + k2*rho^4 + k3*rho^6;
    
    mdu = mu*distortion;
    mdv = mv*distortion;
    
    % add tangential distortion
    mdu = mdu + 2*p1*mdu*mdv + p2*(rho^2 + 2*mdu^2);
    mdv = mdv + p1*(rho^2 + 2*mdv^2) + 2*p2*mdu*mdv;
    
    % transform to image frame
    umodel = fx*mdu + cx;
    vmodel = fy*mdv + cy;
    
    % compute error
    
    error = sqrt((u - umodel)^2 + (v - vmodel)^2);
    errors(i) = error;
end

%
% equivalent unified fisheye camera parameter 
% 393.2787  393.1590    1.2077  320.0035  240.0047   -0.4230    0.1826   -0.0255   -0.0000   -0.0000









