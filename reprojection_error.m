function cost = reprojection_error(x, pixels, rays, nPoints)

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

cost = zeros(nPoints, 1);
for i = 1:nPoints
    u = pixels(1, i); v = pixels(2, i);
    xs = rays(1, i); ys = rays(2, i); zs = rays(3, i);
    
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
    
    % compute reprojection error
    cost(i) = sqrt((u - umodel)^2 + (v - vmodel)^2);
end
end