function [x_3d, Maincost] = IterRecon_PairProd_TV_FISTA (ForBack, g, gamma, mu, ImageSize)
% CT Iterative Reconstruction using BM3D Regularization and FISTA algorithm.
% Solves the optimization (1/2)(||A*W*x-g||_2)^2+gamma*||D*x||_1^{mu}.
% g are the Line Integral images. x is the 3D reconstructed image, or the optimization variable.
% AWx are a list of projection images, which are of the same dimension as g.
% W is a matrix to deal with boundary condition problems.
fprintf('\n\n\n*******Iterative Reconstruction with TV Regularization*******\n\n');

Forward = ForBack.applyFP;
Backward = ForBack.applyBP;

% Initialize the input image
g(g<0)=0;
g(logical(isnan(g)))=0;
g = g(:);

% Initialization.
N = prod(ImageSize);
xk = zeros(N,1); vk = xk;
tk = 1e-06; % Initial step sizes.
r1 = 0.8; r2 = 0.1;

maxIter = 50; count = 1;
Maincost = zeros(maxIter,1);

% Iterative loop
fprintf('Begin Optimization...\n\n');
tic

while (count<=maxIter)
    t = tk/r1;
    subiteration = 0;
    if count == 2
        r2 = 0.5;
    end
    while 1 % Lipschitz Gradient Condition
        if(count == 1)
            theta = 1;
        else
            r = tk/t;
            theta = thetak*(sqrt(4*r+thetak^2) - thetak)/(2*r);
        end
        
        y = (1-theta)*xk + theta*vk;
        
        FPy = Forward(y);%
        DataDiff_y = FPy-g;
        BPz1 = Backward(DataDiff_y); 
        DVP_y = DVP_Dx_2D (mu, reshape(y,ImageSize));
        delta_fy = BPz1(:) + gamma/mu*(DVP_y(:));
        
        x = max(0,y-t*delta_fy);
        
        % Compute f(y)
        Dy = applyD2D (reshape(y,ImageSize));
        Huber_Dy_in = 1/(2*mu)*(Dy).^2;
        Huber_Dy_out = abs(Dy)-mu/2;
        Huber_Dy = Huber_Dy_in;
        Huber_Dy(abs(Dy)>mu) = Huber_Dy_out(abs(Dy)>mu);
        fy = 1/2*sum(DataDiff_y(:).^2) + gamma*sum(Huber_Dy(:));
        
        UpperBound_x = fy + sum(delta_fy.*(x-y)) + 1/(2*t)*sum((x-y).^2);
        
        % Compute f(x)
        FPx = Forward(x);    % Para Version
        DataDiff_x = FPx-g;
        
        Dx = applyD2D (reshape(x,ImageSize));
        Huber_Dx_in = 1/(2*mu)*(Dx).^2;
        Huber_Dx_out = abs(Dx)-mu/2;
        Huber_Dx = Huber_Dx_in;
        Huber_Dx(abs(Dx)>mu) = Huber_Dx_out(abs(Dx)>mu);
        
        fx = 1/2*sum(DataDiff_x(:).^2) + gamma*sum(Huber_Dx(:));
        
        if fx <= UpperBound_x
            break
        end
        
        t = r2*t;
        subiteration = subiteration+1;
    end
    
    tk = t;
    thetak = theta;
    vk = xk + 1/theta*(x - xk);
    xk = x;
    
    Maincost(count) = fx;
    
    if mod(count,5) == 0
        opttime=toc;
        figure(3);semilogy(Maincost,'r');hold on;title(['TV iteration:' num2str(count) ', gamma: ' num2str(gamma)]); pause(0.01);
        x_3d = reshape(x,ImageSize);
        figure(6);imshow(x_3d(:,:,ceil(end/2)), []);title(['TV iteration:' num2str(count) ', gamma: ' num2str(gamma)]); pause(0.01);
%          figure(7);imshow(squeeze(x_3d(:,ceil(end/2),:)),[]);
    end
    count = count+1;
end

fprintf('\n\nEnd Optimization.');
fprintf('\n\n*******Iterative Reconstruction with TV Regularization*******\n\n');

x_3d = reshape(x,ImageSize);

end


function DVP_x = DVP_Dx_2D (mu, x)
D1x=cat(1,diff(x,1,1),zeros(1,size(x,2),size(x,3)));
D2x=cat(2,diff(x,1,2),zeros(size(x,1),1,size(x,3)));

z1=max(-mu,min(mu,D1x));
DTransz= cat(1, -z1(1,:,:), z1(1:end-2,:,:) - z1(2:end-1,:,:), z1(end-1,:,:));
z2=max(-mu,min(mu,D2x));
DVP_x = DTransz + cat(2, -z2(:,1,:), z2(:,1:end-2,:) - z2(:,2:end-1,:), z2(:,end-1,:));

end
