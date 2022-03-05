function [xfbp] = em_fbp_QL(sg, ig, proj)
% arg.kernel = ones(3,1)/3;
% if any(size(arg.kernel) ~= 1)
%     proj = conv2(proj, arg.kernel, 'same'); % filter
% end

%% filter the sino
sino_filt = fbp2_sino_filter('flat', proj,'ds', sg.dr, 'window', '');
% sino_filt = proj;
sino_filt = single(sino_filt);

%% Backprojection
% trick: extra zero column saves linear interpolation indexing within loop!
nb = size(sino_filt,1); % # of radial bins
if nb ~= sg.nb, fail 'nb size', end
sino_filt(end+1,:) = 0;

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
xfbp = 0;
for ia=1:sg.na
    ticker(mfilename, ia, sg.na)
    
    rr = xc * cang(ia) + yc * sang(ia); % [np,1]
    rr = rr / sg.d + sg.w + 1; % unitless bin index, +1 because matlab
        
    % linear interpolation:
    il = floor(rr); % left bin
    wr = rr - il; % left weight
    wl = 1 - wr; % right weight
    xfbp = xfbp + wl .* sino_filt(il, ia) + wr .* sino_filt(il+1, ia);
    %     test = wl .* sino(il, ia) + wr .* sino(il+1, ia);
    %     figure(1);imshow([embed(wl, mask)/max(wl) embed(wr, mask)/max(wr) embed(test, mask)/max(test(:)) embed(img, mask)/max(img(:))])
end

% img = (deg2rad(sg.orbit) / (sg.na/ia_skip)) * embed(img, mask);
xfbp = pi / (sg.na) * embed(xfbp, mask); % 2008-10-14

xfbp = xfbp .* ig.mask;

