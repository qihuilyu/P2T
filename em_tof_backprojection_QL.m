function [xfbp] = em_tof_backprojection_QL(sg, ig, sino3D)
% arg.kernel = ones(3,1)/3;
% if any(size(arg.kernel) ~= 1)
%     proj = conv2(proj, arg.kernel, 'same'); % filter
% end

[nd,~,na] = size(sino3D);

%% Backprojection
% trick: extra zero column saves linear interpolation indexing within loop!
nb = size(sino3D,1); % # of radial bins
if nb ~= sg.nb, fail 'nb size', end
sino3D(end+1,end+1,:) = 0;

[xc yc] = ndgrid(ig.x, ig.y);
rr = sqrt(xc.^2 + yc.^2); % [nx ny]
rmax = ((sg.nb-1)/2-abs(sg.offset)) * sg.d;
mask = ig.mask;
mask = mask & (rr < rmax);
xc = xc(mask(:)); % [np] pixels within mask
yc = yc(mask(:));

cang = cos(sg.ar);
sang = sin(sg.ar);

% loop over each projection angle
xfbp = 0; sz = size(sino3D); numxc = numel(xc);
for ia=1:sg.na
    ticker(mfilename, ia, sg.na)
    
    rr = xc * cang(ia) + yc * sang(ia); % [np,1]
    rr = rr / sg.d + sg.w + 1; % unitless bin index, +1 because matlab
    
    rd = -xc * sang(ia) + yc * cang(ia); % [np,1]
    rd = rd / sg.d + sg.w + 1; % unitless bin index, +1 because matlab
        
    % linear interpolation:
    il = floor(rr); % left bin
    wr = rr - il; % left weight
    wl = 1 - wr; % right weight
    
    % linear interpolation:
    ild = floor(rd); % lower bin
    wlo = rd - ild; % lower weight
    wup = 1 - wlo; % upper weight
    
    iavec = ia*ones(numxc,1);
    indlol = sub2ind(sz,il,ild,iavec);
    indlor = indlol + 1;
    indupl = sub2ind(sz,il,ild+1,iavec);
    indupr = indupl + 1;
    xfbp = xfbp + wlo.* wl.*sino3D(indlol) + wlo.* wr.*sino3D(indlor) ...
        + wup.*wl.*sino3D(indupl) + wup.* wr.*sino3D(indupr);
    %     test = wl .* sino(il, ia) + wr .* sino(il+1, ia);
    %     figure(1);imshow([embed(wl, mask)/max(wl) embed(wr, mask)/max(wr) embed(test, mask)/max(test(:)) embed(img, mask)/max(img(:))])
end

% img = (deg2rad(sg.orbit) / (sg.na/ia_skip)) * embed(img, mask);
xfbp = pi / (sg.na) * embed(xfbp, mask); % 2008-10-14

xfbp = xfbp .* ig.mask;

