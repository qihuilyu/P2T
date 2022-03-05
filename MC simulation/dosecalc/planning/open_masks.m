function [ masks ] = open_masks( h5file, order, verbose )
%{
This function is used to read a list of dense binary masks (rank 3 tensors)
from a .mask file (hdf5). Masks are populated as a collection of '0', '1'
values and expressed as "Logical" tensors.

INPUTS:
    h5file  (str):   full path to .mask (hdf5) file produced by dose calculation engine
    order (str):     reorder output array into ['zyx'(default), 'xyz', 'yxz']
    verbose (bool):  print verbosely; use "2" for extra verbosity

OUTPUTS:
    masks ( {struct(), struct(), ...} ): cell array of struct objects, each containing the fields:
        masks{i}.name (str): name of ROI
        masks{i}.mask (rank 3 tensor - logical): dense boolean volume


Author: Ryan Neph
Revisions:
    - 28 Sept. 2017: Initial
    - 02 Oct. 2017:  Fixed bug in 1d-index that capped at 65535 due to
        matlab data representation
    - 10 May  2020: fixed issue identifying full array size when inexact field matching was used
%}

%% Check args
if nargin < 1
    error('Please provide the argument "h5file"');
elseif nargin > 3
    error('Too many arguments provided');
end
if nargout > 1
    error('Too many output arguments provided')
end
if nargin < 2
    order = 'zyx';
else
    if ischar(order) && ~(strcmp(order,'xyz') || strcmp(order,'yxz') || strcmp(order,'zyx'))
        error('order must be one of ["xyz","yxz","zyx"]');
    end
end
if nargin < 3
    verbose = false;
else
    verbose = int8(verbose);
end

LogicalStr = {'false', 'true', 'extra'};
if verbose
    fprintf('===========================================\n');
    fprintf('==     ROI MASK DATA PARSER (MATLAB)     ==\n');
    fprintf('===========================================\n');
    fprintf('  options:   verbose:   %s\n', LogicalStr{verbose+1});
    fprintf('             order:     %s\n', order);
    fprintf('===========================================\n');
end

%% get mask names
metainfo = h5info(h5file);
nmasks = numel(metainfo.Groups);
paths = cell(nmasks, 1);
indices = zeros(nmasks, 1);
for i=1:nmasks
    paths{i} = metainfo.Groups(i).Name;
    attrs = metainfo.Groups(i).Attributes;
    indices(i) = getAttributeByName(attrs, 'index');
end

% sort by index
[~, ord] = sort(indices, 'ascend');
paths = paths(ord);

%% parse masks in indexed order
masks = cell(nmasks, 1);
for i=1:nmasks
    masks{i} = struct();
    info = h5info(h5file, paths{i});
    attrs = info.Attributes;
    masks{i}.name = getAttributeByName(attrs, 'name');

    % get mask metadata
    arrayprops = getGroupByName(info, 'ArrayProps', false);
    attrs = arrayprops.Attributes;
    % tuples are ordered as XYZ in metadata
    dims = getAttributeByName(attrs, 'size')';
    crop_start = getAttributeByName(attrs, 'crop_start')' + 1;
    crop_size = getAttributeByName(attrs, 'crop_size')';

    if verbose>1
        fprintf('Name: "%s"\n', masks{i}.name{1});
        fprintf('  crop_start (xyz): (%d, %d, %d)\n', crop_start(1), crop_start(2), crop_start(3));
        fprintf('  crop_size (xyz):  (%d, %d, %d)\n', crop_size(1), crop_size(2), crop_size(3));
        fprintf('  full dims (xyz):  (%d, %d, %d)\n', dims(1), dims(2), dims(3));
    end

    % load mask data
    data = h5read(h5file, [paths{i},'/mask']);
    data = permute(reshape(data, crop_size), [3 2 1]);
    if order=='zyx'
        % order as ZYX
        dims = fliplr(dims);
        crop_start = fliplr(crop_start);
        crop_size = fliplr(crop_size);
    elseif order=='yxz'
        % order as YXZ
        dims = [dims(2), dims(1), dims(3)];
        crop_start = [crop_start(2), crop_start(1), crop_start(3)];
        crop_size = [crop_size(2), crop_size(1), crop_size(3)];
        data = permute(data, [2 1 3]);
    elseif order=='xyz'
        % order as XYZ
        data = permute(data, [3 2 1]);
    end

    % insert mask data into full tensor
    mask = zeros(dims);
    st = crop_start;
    sz = crop_size;
    mask(st(1):st(1)+sz(1)-1, st(2):st(2)+sz(2)-1, st(3):st(3)+sz(3)-1) = data;

    masks{i}.mask = logical(mask);
    if verbose >1 fprintf('\n'); end
end

if verbose
    fprintf('Discovered %d masks:\n', nmasks);
    for i=1:nmasks
        fprintf(' - %s\n', masks{i}.name{1});
    end
end

end



%% Helper Functions
function [ val ] = getAttributeByName(attrs, name, exact)
    if nargin < 3
       exact = true;
    else
        exact = logical(exact);
    end
    for aa=1:numel(attrs)
        if (exact && strcmp(attrs(aa).Name, name)) || ...
           (~exact && ~isempty(strfind(attrs(aa).Name, name)))
            val = attrs(aa).Value;
            break;
        end
    end

end
function [ grp ] = getGroupByName(loc, name, exact)
    if nargin < 3
       exact = true;
    else
        exact = logical(exact);
    end
    for i=1:numel(loc.Groups)
        if (exact && strcmp(loc.Groups(i).Name, name)) || ...
           (~exact && ~isempty(strfind(loc.Groups(i).Name, name)))
            grp = loc.Groups(i);
            break;
        end
    end
end
