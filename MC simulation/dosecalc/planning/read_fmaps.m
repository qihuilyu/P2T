%% MODES
function dose_data = read_fmaps( h5file, verbose, mode)
%{
This function provides a parser for the HDF5 data structure created by the beamlet-based dose calcluation
    program. Notably, this function accepts as input the path to the hdf5 file and returns a sparsely
    constructed dose coefficient ("A") matrix which has its columns ordered in increasing beam number then
    increasing beamlet number, and its rows ordered in the linear index of the target voxel. This function
    also returns the ordered list of beam number and beamlet number associated with each column such that
    the correspondence is not lost when its time to map fluence weights back to each selected beam.

INPUTS:
    * h5file (str):    full path to .h5 file produced by beamlet-based dose calculation engine
    * verbose (bool):  print verbosely; use "2" for extra verbosity
    * mode (str):      select the type of data parsing to perform (see below for options)

OUTPUTS (as a structure):
    * sparsemat (rank 2 sparse matrix)
        - sparse dose calculation matrix
    * column_labels (cellarray[ struct{beam, beamlet} ])
        - cellarray of structs, each containing beam and beamlet number for the column indexed by the current
          cellarray index
        - only provided in "sparsemat" and "all" modes
    * calc_metadata (struct{meta...})
        - structure containing key:val pairs of calculation metadata from hdf5 file
    * beam_metadata (struct{meta...})
        - structure containing hierarchy of beam and beamlets, associated attributes and fluence map weights used
          in dose coefficient calculation
    * sparsedata (struct{coeffs, lindex...})
        - beam/beamlet hierarchy of original coeffs and lindex sparse arrays
        - this is the raw beamlet dose data in a sparse COO format; non-zero values are encoded as
        - (linearized_volume_index, dose_value) data pairs for each beamlet
          (only provided in "sparsedata" and "all" modes

MODE OPTIONS:
    * 'sparsemat'   -   Sparse matrix, column labels, and metadata (default)
    * 'sparsedata'  -   Raw sparse data (COO format), and metadata (don't build sparse matrix)
    * 'metadata'    -   Only return metadata
    * 'all'         -   Like Sparsemat, but h5 data is read immediately during metadata compilation
                          (before sorting; this is likely slower than "Sparsemat" and should not be preferred)
                          essentially this is "Sparsedata" with the additional step of building the sparse matrix

-----------------
Author: Ryan Neph
Revisions:
    - 09 Aug.  2017 [v1.0]:   Initial
    - 28 Sept. 2017 [v1.2]:   type check arguments
    - 15 Mar.  2018 [v1.3]:   support for new output format that supports dataset linking (in dosecalc v0.7.0+)
    - 16 Mar.  2018 [v1.4]:   Fixed for R2016a and below (missing "contains" function)
    - 11 Mar.  2019 [v1.5]:   Redesigned function input/output format for better flexibility; dropped support for old file formats
    - 11 Jun.  2019 [v1.6]:   Fixed beam count check when some can be skipped; replaced double with single quotes to define string
    - 13 Jun.  2019 [v1.7]:   Add support for "skipped" beams (0 beamlets); joined Modes.m into this file
    - 6  Jan.  2020 [v1.8]:   Added support to read beam metadata from FMAPS files
%}
VERSION_MAJOR = '1';
VERSION_MINOR = '7';

% Prepare function result structure
dose_data = struct;

% timers
timer = tic;

global MODE_SPARSEMAT MODE_SPARSEDATA MODE_METADATA MODE_ALL
MODE_SPARSEMAT  = 0;
MODE_SPARSEDATA = 1;
MODE_METADATA   = 2;
MODE_ALL        = 3;

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
    verbose = false; % default
else
    verbose = max(0, min(2, double(verbose) ));
end
if nargin < 3
    MODE = MODE_SPARSEMAT; % default
else
    % mode selection
    switch lower(mode)
        case 'sparsemat'
            MODE = MODE_SPARSEMAT;
        case 'sparsedata'
            MODE = MODE_SPARSEDATA;
        case 'metadata'
            MODE = MODE_METADATA;
        case 'all'
            MODE = MODE_ALL;
        otherwise
            error('Acceptable modes are: ["sparsemat" (default), "sparsedata", "metadata", "all"]');
    end
end

metainfo = h5info(h5file);

%% check filetype/version
ftgroup = getGroupByName(metainfo, 'filetype', true);
if isstruct(ftgroup)
    ftversionmajor = getAttributeByName(ftgroup.Attributes, 'ftversionmajor', true);
    ftversionminor = getAttributeByName(ftgroup.Attributes, 'ftversionminor', true);
    ftmagic = getAttributeByName(ftgroup.Attributes, 'ftmagic', true);

    switch ftmagic
        case 42
            ftname = 'Dose';
        case 43
            ftname = 'Beamlet Dose';
        case 44
            ftname = 'Fmaps';
        otherwise
            ftname = 'unknown';
    end
else
    ftmagic = -1;
    ftname = 'unknown';
    ftversionmajor = 0;
    ftversionminor = 0;
end

if verbose
    LogicalStr = {'false', 'true', 'extra'};
    fprintf('===========================================\n');
    fprintf('==   DOSE CALCULATION DATA PARSER v%s   == \n', [VERSION_MAJOR, '.', VERSION_MINOR]);
    fprintf('===========================================\n');
    fprintf('  filetype: %s (v%d.%d)\n', ftname, ftversionmajor, ftversionminor);
    fprintf('\n')
    fprintf('  options:    verbose:  %s\n', LogicalStr{verbose+1});
    fprintf('              mode:     %s\n', char(mode));
    fprintf('===========================================\n');
    clear LogicalStr;
end

%% read data
vhash = version_hash(ftversionmajor, ftversionminor);
switch ftmagic
    case 43
        ;
    case 44
        MODE = MODE_METADATA;
        fprintf(['File only contains beam metadata. Setting mode to "metadata"\n']);
    otherwise
        fprintf('Invalid filetype: %s. Reading of this filetype is not supported by this function\n', ftname);
        return;
end
result = read_sparse_format(h5file, metainfo, vhash, MODE, verbose);

time = toc(timer);
result.read_time_sec = time;
if verbose
    fprintf('\n');
    fprintf('Full Execution Time: %0.3f secs\n', time);
end

dose_data = result;
end


%% Format Specific Parsing Functions
function result = read_sparse_format(h5file, metainfo, vhash, mode, verbose)
global MODE_SPARSEMAT MODE_SPARSEDATA MODE_METADATA MODE_ALL;
result = struct;

% state_vars
if mode == MODE_SPARSEDATA || mode == MODE_ALL
    keep_spdata = true;
else
    keep_spdata = false;
end

% format compatability test
if vhash <= 0
    error('The file being read uses a legacy format (v%d.%d) that is not supported by this function', ftversionmajor, ftversionminor);
end

%% extract metadata
if verbose
    fprintf('\n');
    disp('EXTRACTING METADATA');
    disp('-------------------');
end

try
    idx = 0;
    for i=numel(metainfo.Groups):-1:1 % usually found at end - faster to reverse iterate
        if strcmp(metainfo.Groups(i).Name,  '/calc_specs')
            idx = i;
            if verbose
                fprintf('  Found calc_meta at index %d\n', i);
            end
            break
        end
    end
    if idx == 0
        error('  /calc_specs was not found. Please check the format of the supplied hdf5 file path. Exiting early');
    end
    meta_attrs = metainfo.Groups(idx).Attributes;
    nattrs = numel(meta_attrs);
    if verbose
        fprintf('  Located %d attributes\n', nattrs);
    end
    calc_meta = struct; %OUTVAR
    for i=1:nattrs
        calc_meta.(meta_attrs(i).Name) = meta_attrs(i).Value;
    end

    % check if data is for reduced matrix or full matrix
    reduced = isfield(calc_meta, 'roi_order');
    if verbose
        if reduced; fprintf('  *Reduced matrix data detected*'); end
        fprintf('\n');
    end
    clear meta_attrs nattrs;
catch exception
    if mode ~= MODE_METADATA
        error(exception)
    end
    calc_meta = struct();
end


%% Sort beams
if verbose
    fprintf('\n');
    disp('DISCOVERING BEAMS/BEAMLETS');
    disp('--------------------------');
end
spdata = struct([]);
beam_meta = struct([]);
NB = 0;  % number of beams discovered
Nbt = 0; % number of beamlets discovered (total)
Nnonzero = 0; % total number of non-zero elements in final matrix
if vhash >= 101
    beamgroup = getGroupByName(metainfo, 'beams', true);
    beammetagroup = getGroupByName(beamgroup, 'metadata', true);
    beamdatagroup = getGroupByName(beamgroup, 'data', true);
else
    beamgroup = metainfo;
    beammetagroup = beamgroup;
    beamdatagroup = beamgroup;
end
clear beamgroup metainfo;
for B=1:(numel(beammetagroup.Groups))
    [~, Bname, ~] = fileparts(beammetagroup.Groups(B).Name);
    if strcontains(Bname, 'beam_')
        NB = NB+1;

        attrs = beammetagroup.Groups(B).Attributes;
        beam_meta(NB).Name = Bname;
        for aa=1:numel(attrs)
            attname = attrs(aa).Name;
            beam_meta(NB).(attname) = attrs(aa).Value;
            if keep_spdata && strcontains(attname, '_uid')
                spdata(NB).(attname) = beam_meta(NB).(attname);
            end
        end

        try
            % Sort Beamlets
            Nb = 0; % number of beamlets discovered (for this beam)
            beamlets = struct([]);
            bspdata = struct([]);
            bgroup = beamdatagroup.Groups(B);
            bgroups = bgroup.Groups;
            if numel(bgroups)<=0
                continue
            end
            for b=1:numel(bgroups)
                [~, bname, ~] = fileparts(bgroups(b).Name);
                if strcontains(bname, 'beamlet_')
                    Nb = Nb+1;
                    Nbt = Nbt+1;
                    attrs = bgroups(b).Attributes;
                    beamlets(Nb).Name = bname;
                    for aa=1:numel(attrs)
                        attname = attrs(aa).Name;
                        beamlets(Nb).(attname) = attrs(aa).Value;
                        if keep_spdata && ~isempty(strfind(attrs(aa).Name, '_uid'))
                            bspdata(Nb).(attname) = beamlets(Nb).(attname);
                        end
                    end
                    if verbose == 2
                        fprintf('  || beam: %4d | beamlet: %4d (%4d) | N_coeffs: %7d ||\n', ...
                            B, b, beamlets(Nb).beamlet_uid, beamlets(Nb).N_coeffs);
                    end
                    % store coeffs to separate data struct
                    if keep_spdata
                        for dd=1:numel(bgroups(b).Datasets)
                            dsname = bgroups(b).Datasets(dd).Name;
                            if keep_spdata
                                if vhash >= 101
                                    dspath = ['/beams/data/', Bname, '/', bname, '/', dsname];
                                else
                                    dspath = ['/', Bname, '/', bname, '/', dsname];
                                end
                                bspdata(Nb).(dsname) = h5read(h5file, dspath); % repeated h5read() call, wasteful
                            end
                            % error check
                            if numel(bspdata(Nb).(dsname)) ~= beamlets(Nb).N_coeffs
                                error('  Number of elements in "%s" (%d) doesn"t match metadata count (%d)', dsname, numel(beamlets(Nb).(dsname)), beamlets(Nb).N_coeffs);
                            end
                        end %for dd
                    end %keep_spdata
                    Nnonzero = Nnonzero + beamlets(Nb).N_coeffs;
                end %if isempty
            end %for b
            % error check
            if beam_meta(NB).N_beamlets ~= Nb
                error('  Number of beamlets found (%d) doesn"t match metadata count (%d)', Nb, beam_meta(NB).N_beamlets)
            end
            % apply sorting
            [~, beamletord] = sort([beamlets(:).beamlet_uid], 'ascend');
            beamlets = beamlets(beamletord);
            beam_meta(NB).beamlets = beamlets;
            if keep_spdata
                bspdata = bspdata(beamletord);
                spdata(NB).beamlets = bspdata;
            end
            clear beamlets beamletord;
        catch exception
            if mode ~= MODE_METADATA
                error(exception);
            end
        end
    end %if isempty
end %for B
clear beammetagroup beamdatagroup bgroup bgroups Bname bname attname aa;

 % error check
 if mode ~= MODE_METADATA
     if calc_meta.N_beams ~= NB
         warning('  Number of beams found (%d) doesn"t match metadata count (%d)', NB, calc_meta.N_beams);
     end
 end
 if verbose
     fprintf('  Discovered %d beams\n', NB);
     fprintf('  Discovered %d beamlets (total)\n', Nbt);
 end
 % apply sorting
[~, beamord] = sort([beam_meta(:).beam_uid], 'ascend');
beam_meta = beam_meta(beamord);
if keep_spdata
    spdata = spdata(beamord);
end
clear beamord;


%% Early exits
% Begin packing result
result.calc_metadata = calc_meta;
result.beam_metadata = beam_meta;

if mode == MODE_SPARSEDATA
    result.sparsedata = spdata;
    return
elseif mode == MODE_METADATA
    return
end

%% construct col_labels
col_labels = zeros(Nbt, 2, 'int16');
iii = 0;
for B=1:numel(beam_meta)
    this_beam = beam_meta(B);
    for b=1:numel(this_beam.beamlets)
        iii = iii+1;
        this_beamlet = this_beam.beamlets(b);
        col_labels(iii, :) = [this_beam.beam_uid, this_beamlet.beamlet_uid];
    end
end
result.column_labels = col_labels;

%% Construct sparse coefficient matrix
if verbose
    fprintf('\n');
    disp('CONSTRUCTING SPARSE MATRIX');
    disp('--------------------------');
end
if Nnonzero <= 0
    warning('No non-zero elements could be located. Returning empty matrix and metadata');
    return;
end
if reduced
    % A-matrix has been reduced using PTV+OAR masks
    Nrows = sum(calc_meta.row_block_capacities);
    format = 'reduced (A-matrix)';
else
    Nrows = prod(calc_meta.full_dicom_size);
    format = 'full (M-matrix)';
end

Ncols = Nbt;
% 'single' sparse matrices are not yet supported as of R2017a :(
r = ones(Nnonzero, 1, 'double');
c = ones(Nnonzero, 1, 'double');
v = zeros(Nnonzero, 1, 'double');
ptr = 1;      % marks next insertion location
col_ptr = 1;  % marks next sparse matrix column location
for B=1:numel(beam_meta)
    Bname = beam_meta(B).Name;
    for b=1:numel(beam_meta(B).beamlets)
        bname = beam_meta(B).beamlets(b).Name;
        if vhash >= 101
            dspath = ['/beams/data/', Bname, '/', bname, '/'];
        else
            dspath = ['/', Bname, '/', bname, '/'];
        end
        Ninsert = beam_meta(B).beamlets(b).N_coeffs;
        if verbose == 2
            fprintf('|| beam: %4d | beamlet: %4d (%4d) | column: %4d | inserting: %7d | at: %10d ||\n', ...
                B, b, beam_meta(B).beamlets(b).beamlet_uid, col_ptr, Ninsert , ptr);
        end

        if keep_spdata
            r(ptr:ptr+Ninsert-1, 1) = spdata(B).beamlets(b).lindex + 1;
            c(ptr:ptr+Ninsert-1, 1) = col_ptr;
            v(ptr:ptr+Ninsert-1, 1) = spdata(B).beamlets(b).coeffs;
        else
            % increment r by +1 (matlab 1-indexing, data 0-indexing)
            r(ptr:ptr+Ninsert-1, 1) = h5read(h5file, [dspath, 'lindex']) + 1;
            c(ptr:ptr+Ninsert-1, 1) = col_ptr;
            v(ptr:ptr+Ninsert-1, 1) = h5read(h5file, [dspath, 'coeffs']);
        end
        ptr = ptr + Ninsert;
        col_ptr = col_ptr + 1;
    end %for b
end %for B
sparsemat = sparse(r, c, v, Nrows, Ncols); %OUTVAR

[ssize, sunit] = storage_size(sparsemat);
if verbose
    fprintf('  Constructed sparse matrix: \n');
    fprintf('     -- size:    (%d x %d)\n', Nrows, Ncols);
    fprintf('     -- nonzero:  %d\n', Nnonzero);
    fprintf('     -- density:  %0.2e%%\n', (Nnonzero)*100/(Nrows*Ncols));
    fprintf('     -- storage:  %0.3f %s\n', ssize, sunit);
    fprintf('     -- format:   %s\n', format);
end

sparse_info = struct();
sparse_info.nnz = Nnonzero;
sparse_info.density = double(Nnonzero)/(Nrows*Ncols);
sparse_info.storage_size = ssize;
sparse_info.storage_unit = sunit;
result.sparsemat_info = sparse_info;
result.sparsemat = sparsemat;
if keep_spdata
    result.sparsedata = spdata;
end
return;
end

%% Helper Functions
function [size, unit] = storage_size(var)
% Format storage size of sparsemat
bytes = whos('var');
bytes = bytes.bytes;
if bytes < 1024
    size = bytes;
    unit = 'B';
elseif bytes < 1048576
    size = bytes/1024;
    unit = 'KB';
elseif bytes < 1073741824
    size = bytes/1048576;
    unit = 'MB';
else
    size = bytes/1073741824;
    unit = 'GB';
end
end
function vhash = version_hash(major, minor)
% Produce standardized format version hash for numeric comparison
vhash = 100*major + minor;
end
function val = getAttributeByName(attrs, name, exact)
% safely read H5 attribute value by name
if nargin < 3
    exact = false;
else
    exact = logical(exact);
end
for aa=1:numel(attrs)
    if (exact && strcmp(attrs(aa).Name, name)) || ...
            (~exact && strcontains(attrs(aa).Name, name))
        val = attrs(aa).Value;
        break;
    end
end
end
function grp = getGroupByName(loc, name, exact)
% return H5 group object by name
grp = false;

if nargin < 3
    exact = false;
else
    exact = logical(exact);
end
for i=1:numel(loc.Groups)
    [~, qname, ~] = fileparts(loc.Groups(i).Name);
    if (exact && strcmp(qname, name)) || ...
            (~exact && strcontains(qname, name))
        grp = loc.Groups(i);
        break;
    end
end
end
function out = strcontains(fullstr, sub)
% portable version of contains(str1, str2) for matlab before 2016b
if ~isempty(strfind(fullstr, sub)) %#ok<*STREMP>
    out=true;
else
    out=false;
end
end


