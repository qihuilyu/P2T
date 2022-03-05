import math

from PIL import Image
import numpy as np
from numpy.ma import masked_array
from matplotlib import colors, cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2

FIG_DPI = 150

class MidpointNormalize(colors.Normalize):
    """custom colormap with two linear ranges as per https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges"""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def figure2array(fig):
    fig.canvas.draw()
    buf, (width, height) = fig.canvas.print_to_buffer()
    # buffer is rgba
    d = np.frombuffer(buf, dtype=np.uint8)
    d = d.reshape((height, width, 4))
    d = d[None, :, :, :]  # TF2 requires a batch axis in front
    plt.close(fig)
    return d

def save_figure_array(figarr, f):
    """save array representing colored figure (produced by figure2array) to file"""
    # figarr by default has shape [1,H,W,3] for compat. with tensorboard
    Image.fromarray(figarr[0]).save(f)


ignored_cmap = None
def get_ignored_cmap():
    global ignored_cmap
    if ignored_cmap is None:
        #  ignored_cmap = colors.LinearSegmentedColormap('ignored', {'red': ((0,0.1,0.1), (1,0.1,0.1)), 'green': ((0,0.5,0.5), (1,0.5,0.5)), 'blue': ((0,0.1,0.1), (1,0.1,0.1))})
        ignored_cmap = colors.LinearSegmentedColormap('ignored', {'red': ((0,0.325,0.325), (1,0.325,0.325)), 'green': ((0,0.325,0.325), (1,0.325,0.325)), 'blue': ((0,0.325,0.325), (1,0.325,0.325))})
    return ignored_cmap

def create_volume_dose_figure(arr, col_labels=[], dpi=FIG_DPI, cmap='viridis', own_scale=False, return_fig=False, ndiff_cols=1):
    """create figure of axes rows mapped to each slice in numpy array [Rows, Cols, H, W]
    with each row containing imshow() instances with matched value limits (vmin, vmax) according
    to data (min, max)
    """
    nrows, ncols, H, W = arr.shape
    axwidth_inches = 1.25
    width_ratios=[1.0]*ncols + [0.25]*2
    figwidth = axwidth_inches*np.sum(width_ratios)
    figheight = axwidth_inches/W*H*nrows/0.98
    fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi)
    spec = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols+2,
        width_ratios=width_ratios,
        wspace=0,
        hspace=0,
        left=0.0, right=1.0,
        bottom=0.0, top=0.98,
        figure=fig,
    )

    # annotation settings
    annotate_fontsize = 5
    annotate_margin = 0.03 # as fraction of imshow axes

    # create axes and set data
    row_minmax = []
    row_diff_absmax = []
    for row in range(nrows):
        rowarr = arr[row]
        if ndiff_cols>0:
            vmin, vmax = np.amin(rowarr[:-ndiff_cols]), np.amax(rowarr[:-ndiff_cols])
            diffmin, diffmax = np.amin(rowarr[-ndiff_cols:]), np.amax(rowarr[-ndiff_cols:])
            diffabsmax = max(abs(diffmin), abs(diffmax))
            row_diff_absmax.append(diffabsmax)
        else:
            vmin, vmax = np.amin(rowarr), np.amax(rowarr)
        row_minmax.append((vmin, vmax))
        for col in range(ncols):
            if col>=ncols-ndiff_cols:
                # logarithmic cmap with diverging (difference map)
                #  this_norm      = colors.SymLogNorm(vmin=-diffabsmax, vmax=diffabsmax, linthresh=0.01, linscale=0.01)
                this_norm = colors.Normalize(vmin=-diffabsmax, vmax=diffabsmax)
                this_cmap      = 'RdBu_r'
                this_fontcolor = 'black'
            else:
                # linear cmap with default colors
                this_norm      = colors.Normalize(vmin=vmin, vmax=vmax)
                this_cmap      = cmap
                this_fontcolor = 'white'

            # draw array as image
            cellarr = rowarr[col]
            ax = fig.add_subplot(spec[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(cellarr,
                      interpolation='none', aspect='equal',
                      cmap=this_cmap,
                      norm=this_norm if not own_scale else None)

            # add image min/max as annotation
            margin = 0.03
            fmt = '{:0.2f}'
            cellmin, cellmax = np.amin(cellarr), np.amax(cellarr)
            ax.text(margin, 1.0-margin, fmt.format(cellmax),
                    fontsize=annotate_fontsize,
                    color=this_fontcolor,
                    horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes)
            ax.text(margin, margin, fmt.format(cellmin),
                    fontsize=annotate_fontsize,
                    color=this_fontcolor,
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=ax.transAxes)


            # add column headings
            if row==0 and col < len(col_labels):
                ax.text(0.5, 1.01, col_labels[col],
                        horizontalalignment='center', verticalalignment='bottom',
                        transform=ax.transAxes)

    # add shared colorbar (standard)
    cbar_ax = fig.add_subplot(spec[:, -2])
    fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap='viridis'),
        cax=cbar_ax,
        ticks=[],
    )
    if ndiff_cols > 0:
        # add shared colorbar (difference)
        cbar_ax2 = fig.add_subplot(spec[:, -1])
        fig.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu_r'),
            cax=cbar_ax2,
            ticks=[],
        )

    # add row-wise vlimit labels to colorbar
    for row in range(nrows):
        ypos_high = 1.0 - (float(row+margin)/nrows)
        ypos_low  = 1.0 - (float(row+1-margin)/nrows)
        row_min, row_max = row_minmax[row]
        cbar_ax.text(0.5, ypos_high, '{:0.2f}'.format(row_max),
                     fontsize=annotate_fontsize,
                     horizontalalignment='center', verticalalignment='top',
                     transform=cbar_ax.transAxes)
        cbar_ax.text(0.5, ypos_low, '{:0.2f}'.format(row_min),
                     fontsize=annotate_fontsize,
                     horizontalalignment='center', verticalalignment='bottom',
                     transform=cbar_ax.transAxes)
        if ndiff_cols > 0:
            row_diffabsmax = row_diff_absmax[row]
            cbar_ax2.text(0.5, ypos_high, '{:0.2f}'.format(row_diffabsmax),
                         fontsize=annotate_fontsize,
                         horizontalalignment='center', verticalalignment='top',
                         transform=cbar_ax2.transAxes)
            cbar_ax2.text(0.5, ypos_low, '{:0.2f}'.format(-row_diffabsmax),
                         fontsize=annotate_fontsize,
                         horizontalalignment='center', verticalalignment='bottom',
                         transform=cbar_ax2.transAxes)

    if return_fig:
        return fig
    else:
        return figure2array(fig)


def make_image_figure(colorbar=True, dpi=FIG_DPI):
    """makes standard figure with large imshow axes and colorbar to right"""
    fig = plt.figure(dpi=dpi)
    if colorbar:
        ax_im = fig.add_axes([0, 0, 0.87, 1.0])
        ax_cbar = fig.add_axes([0.89, 0.05, 0.04, 0.9])
    else:
        ax_im = fig.add_axes([0, 0, 1.0, 1.0])
        ax_cbar = None
    return (fig, ax_im, ax_cbar)

def plot_dose(*args, colorbar=True, dpi=FIG_DPI, **kwargs):
    """See _plot_gamma_components for function signature"""
    fig, ax_im, ax_cbar = make_image_figure(colorbar, dpi)
    _plot_dose(ax_im, ax_cbar, *args, **kwargs)
    return figure2array(fig)

def _plot_dose(ax_im, ax_cbar, arr, **kwargs):
    """plots array using imshow with colorbar then converts back to png compatible rgb array"""
    kwargs['cmap'] = kwargs.get('cmap', 'viridis')
    im = ax_im.imshow(arr, interpolation='nearest', **kwargs)
    ax_im.set_axis_off()
    if ax_cbar is not None:
        plt.colorbar(im, ax_cbar)
    return (ax_im, ax_cbar)

def plot_gamma(*args, colorbar=True, dpi=FIG_DPI, **kwargs):
    """See _plot_gamma_components for function signature"""
    fig, ax_im, ax_cbar = make_image_figure(colorbar, dpi)
    _plot_gamma(fig, ax_im, ax_cbar, *args, **kwargs)
    return figure2array(fig)

def _plot_gamma(fig, ax_im, ax_cbar, arr, annotate=None, **kwargs):
    """plots array using imshow with colorbar then converts back to png compatible rgb array"""
    ignored_cmap = get_ignored_cmap()

    im = ax_im.imshow(arr, cmap='RdBu_r', interpolation='nearest', norm=MidpointNormalize(0, 10, 1), **kwargs)
    im1 = masked_array(arr,arr>=0) # shows ignored values
    ax_im.imshow(im1, cmap=ignored_cmap, interpolation='nearest')
    ax_im.set_axis_off()
    if ax_cbar is not None:
        plt.colorbar(im, ax_cbar)
    if annotate:
        ax_im.text(0.02,0.02, str(annotate),
                 fontsize=11,
                 bbox={'facecolor':'white', 'alpha':1.0},
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=fig.transFigure,
                 )
    return (ax_im, ax_cbar)

def plot_gamma_components(*args, colorbar=True, dpi=FIG_DPI, **kwargs):
    """See _plot_gamma_components for function signature"""
    fig, ax_im, ax_cbar = make_image_figure(colorbar, dpi)
    _plot_gamma_components(fig, ax_im, ax_cbar, *args, **kwargs)
    return figure2array(fig)

def _plot_gamma_components(fig, ax_im, ax_cbar, arr_dd, arr_dta, annotate=None, array_spacing=2, **kwargs):
    """plots array using imshow with colorbar then converts back to png compatible rgb array"""
    ignored_cmap = get_ignored_cmap()

    arr = np.concatenate([arr_dd, -1*np.ones((arr_dd.shape[0], array_spacing)), arr_dta], axis=1)

    im = ax_im.imshow(arr, cmap='RdBu_r', interpolation='nearest', norm=MidpointNormalize(0, 10, 1), **kwargs)
    im1 = masked_array(arr,arr>=0) # shows ignored values
    ax_im.imshow(im1, cmap=ignored_cmap, interpolation='nearest')
    ax_im.set_axis_off()

    # annotate with component specific passing percentages
    try: dd_passing = (np.count_nonzero(arr_dd<=1)-np.count_nonzero(arr_dd<0))/np.count_nonzero(arr_dd>=0)   # passing score for our purpose
    except: dd_passing = np.nan
    ax_im.text(0.02,0.02, 'dd: {:0.2f}%'.format(dd_passing*100),
             fontsize=9,
             bbox={'facecolor':'white', 'alpha':1.0},
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax_im.transAxes,
             )
    try: dta_passing = (np.count_nonzero(arr_dta<=1)-np.count_nonzero(arr_dta<0))/np.count_nonzero(arr_dta>=0)   # passing score for our purpose
    except: dta_passing = np.nan
    ax_im.text(0.52,0.02, 'dta: {:0.2f}%'.format(dta_passing*100),
             fontsize=9,
             bbox={'facecolor':'white', 'alpha':1.0},
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax_im.transAxes,
             )


    if ax_cbar is not None:
        plt.colorbar(im, ax_cbar)
    if annotate:
        ax_im.text(0.02,0.02, str(annotate),
                 fontsize=11,
                 bbox={'facecolor':'white', 'alpha':1.0},
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=fig.transFigure,
                 )
    return (ax_im, ax_cbar)

def plot_profile(arr_pred, arr_true=None, annotate=None, dpi=FIG_DPI, **kwargs):
    """plots line profiles at various depths then converts back to png compatible rgb array"""
    idx = [0.1, 0.3, 0.5, 0.7, 0.9]
    color = ['green', 'red', 'yellow', 'skyblue', 'blue']
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    x_axis = [i for i in range(arr_pred.shape[1])]
    for j in range(len(idx)):
        sliceidx = int(idx[j]*arr_pred.shape[0])
        dose_pred = arr_pred[sliceidx,:].tolist()
        ax.plot(x_axis,dose_pred,color[j],label='profile_at_%d_pixel'%sliceidx, **kwargs)
        if arr_true is not None:
            dose_true = arr_true[sliceidx,:].tolist()
            ax.plot(x_axis,dose_true,color[j],linestyle=':',label=None, **kwargs)

    plt.legend()
    ax.set_ylabel('dose')
    if annotate:
        ax.text(0.02,0.02, str(annotate),
                 fontsize=11,
                 bbox={'facecolor':'white', 'alpha':1.0},
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=fig.transFigure,
                 )
    return figure2array(fig)

def plot_gamma_scatter(*args, dpi=FIG_DPI, **kwargs):
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    _plot_gamma_scatter(fig, ax, *args, **kwargs)
    fig.tight_layout()
    return figure2array(fig)

def _plot_gamma_scatter(fig, ax, arr_dd, arr_dta, arr_gamma, dd_thresh, dta_thresh, **kwargs):
    """place voxels on scatter-plot based on coordinates in dd-dta space"""
    select = np.logical_and(arr_gamma>=0, np.isfinite(arr_gamma))
    dd_flat = np.ravel(arr_dd[select])*dd_thresh*100
    dta_flat = np.ravel(arr_dta[select])*dta_thresh
    gamma_flat = np.ravel(arr_gamma[select])

    scat = ax.scatter(dd_flat, dta_flat, s=4, marker='o', color='black')#, c=gamma_flat, cmap=ignored_cmap, norm=MidpointNormalize(0,10,1))
    dd_max = np.max(dd_flat)
    dta_max = np.max(dta_flat)
    axis_buffer = 0.01
    ax.set_xlim([-axis_buffer*dd_max, dd_max+axis_buffer*dd_max])
    ax.set_ylim([-axis_buffer*dta_max, dta_max+axis_buffer*dta_max])
    ax.set_xlabel('Percent dose difference')
    ax.set_ylabel('Distance to agreement (mm)')

    # criteria lines
    lineargs = {'linewidth': 1, 'linestyle': '-', 'color': 'black'}
    if dd_max > dd_thresh*100:
        ax.add_line(Line2D(ax.get_xlim(), [dta_thresh, dta_thresh], **lineargs))
    if dta_max > dta_thresh:
        ax.add_line(Line2D([dd_thresh*100, dd_thresh*100], ax.get_ylim(), **lineargs))

    # text annotation
    try: gamma_passing = (np.count_nonzero(arr_gamma<=1)-np.count_nonzero(arr_gamma<0))/np.count_nonzero(arr_gamma>=0)   # passing score for our purpose
    except: gamma_passing = np.nan
    try: dd_passing = (np.count_nonzero(arr_dd<=1)-np.count_nonzero(arr_dd<0))/np.count_nonzero(arr_dd>=0)   # passing score for our purpose
    except: dd_passing = np.nan
    try: dta_passing = (np.count_nonzero(arr_dta<=1)-np.count_nonzero(arr_dta<0))/np.count_nonzero(arr_dta>=0)   # passing score for our purpose
    except: dta_passing = np.nan
    nautofail = np.count_nonzero(np.isinf(arr_gamma))/np.count_nonzero(arr_gamma>=0)
    ax.text(1.0, 1.0, 'gamma: {:0.2f}%\ndd: {:0.2f}%\ndta: {:0.2f}%\ninf: {:0.2f}%'.format(gamma_passing*100, dd_passing*100, dta_passing*100, nautofail*100),
             fontsize=11,
             bbox={'facecolor':'white', 'alpha':1.0},
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes,
             )
    return ax


def plot_gamma_summary(arr_dd, arr_dta, arr_gamma, dd_thresh, dta_thresh, colorbar=True, annotate=None, dpi=FIG_DPI*2, **kwargs):
    fig = plt.figure(dpi=dpi)
    gs_l = gridspec.GridSpec(2, 2, fig, right=0.87)
    gs_r = gridspec.GridSpec(1, 1, fig, left=0.89)
    ax_im_gamma = fig.add_subplot(gs_l[0,0])
    ax_im_scatter = fig.add_subplot(gs_l[0,1])
    ax_im_components = fig.add_subplot(gs_l[1,:])
    ax_cbar = fig.add_subplot(gs_r[:,:])

    _plot_gamma(fig, ax_im_gamma, ax_cbar, arr_gamma)
    _plot_gamma_scatter(fig, ax_im_scatter, arr_dd, arr_dta, arr_gamma, dd_thresh, dta_thresh)
    _plot_gamma_components(fig, ax_im_components, None, arr_dd, arr_dta)

    fig.tight_layout()
    return figure2array(fig)


def register_custom_scalars_layout(writer):
    """define custom plotting in 'Custom Scalars' tab of TensorBoard"""
    layout_summary = summary_lib.custom_scalar_pb(
        layout_pb2.Layout(category=[
            layout_pb2.Category(
                title="all",
                chart=[
                    layout_pb2.Chart(
                        title='loss',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'train/loss',r'eval/loss/test', r'eval/loss/train'],
                        )
                    ),
                    layout_pb2.Chart(
                        title='eval-avg_gammapass/0.1mm_0.1%',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'eval-avg_gammapass/0.1mm_0.1%/.*'],
                        )
                    ),
                    layout_pb2.Chart(
                        title='eval-avg_gammapass/0.2mm_0.2%',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'eval-avg_gammapass/0.2mm_0.2%/.*'],
                        )
                    ),
                    layout_pb2.Chart(
                        title='eval-avg_gammapass/0.5mm_0.5%',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'eval-avg_gammapass/0.5mm_0.5%/.*'],
                        )
                    ),
                    layout_pb2.Chart(
                        title='eval-avg_gammapass/1.0mm_1.0%',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'eval-avg_gammapass/1.0mm_1.0%/.*'],
                        )
                    ),
                    layout_pb2.Chart(
                        title='eval-avg_gammapass/2.0mm_2.0%',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'eval-avg_gammapass/2.0mm_2.0%/.*'],
                        )
                    ),
                    layout_pb2.Chart(
                        title='MSE',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'.*mse.*'],
                        )
                    ),
                ],
            ),
        ])
    )
    writer.add_summary(layout_summary)




def tile(array_list, perrow, square=False, pad_width=5, pad_intensity=1000):
    """Takes a list of arrays and number of images per row and constructs a tiled array for margin-less
    visualization

    Args:
        array_list    -- list of np.ndarrays to be tiled in row-major order
        perrow        -- integer specifying number of images per row

    Optional Args:
        square        -- Try to make length and width equal by tiling vertical columns side-by-side
        pad_width     -- # columns between vertical tiling columns
        pad_intensity -- # intensity value of padding cells

    Returns:
        numpy matrix/2dArray
    """
    # setup
    if (not isinstance(array_list, list)):
        array_list_old = array_list
        ndims = len(array_list_old.shape)
        if (ndims == 3):
            array_list = []
            array_list_old_2dshape = (array_list_old.shape[1], array_list_old.shape[2])
            for i in range(array_list_old.shape[0]):
                array_list.append(array_list_old[i, :, :].reshape(array_list_old_2dshape))
        elif (ndims == 2):
            array_list = [array_list_old]
    nimages = len(array_list)
    expect_row_shape = (array_list[0].shape[0], perrow * array_list[0].shape[1])

    # make concatenated rows
    rows_list = []
    this_row_array = None
    for i in range(nimages+1):
        if (i % perrow == 0):
            # add previous row to list
            if (i > 0):
                rows_list.append(this_row_array)
                this_row_array = None
            # start new row
            if i < nimages:
                this_row_array = array_list[i]
        else:
            # add to row
            this_row_array = np.concatenate((this_row_array, array_list[i]), axis=1)

    # extend short rows with zeros
    for i, row in enumerate(rows_list):
        if (row.shape != expect_row_shape):
            extra = np.zeros((expect_row_shape[0], expect_row_shape[1] - row.shape[1]))
            row = np.concatenate((row, extra), axis=1)
            rows_list[i] = row

    # concatenate rows into matrix
    if (square):
        # try to make length and width equal by tiling vertically, leaving a space and continuing in
        # another column to the right
        if (pad_width >= 0):
            pad = pad_width
        else:
            pad = 0
        if (pad_intensity <= 0):
            pad_intensity = 0

        rows = len(rows_list) * expect_row_shape[0]
        cols = expect_row_shape[1]
        # get area, then find side length that will work best
        area = rows * cols
        pref_rows = math.ceil((math.sqrt(area) / expect_row_shape[0]))
        # pref_cols = int(area / (pref_rows * expect_row_shape[0]) / expect_row_shape[1]) + 1

        # construct matrix
        cols_list = []
        this_col_array = []
        for i in range(len(rows_list)+1):
            if (i%pref_rows == 0) or i >= len(rows_list):
                if (i>0):
                    # add previous column to list
                    cols_list.append(this_col_array)
                    if i>= len(rows_list):
                        break

                    if (pad > 0 and i < len(rows_list)-1):
                        # add padding column
                        cols_list.append(pad_intensity * np.ones((pref_rows * expect_row_shape[0], pad)))

                # start new column
                this_col_array = rows_list[i]
            else:
                # add to column
                this_col_array = np.concatenate((this_col_array, rows_list[i]), axis=0)

        # extend short cols with zeros
        for i, col in enumerate(cols_list):
            if (col.shape[0] != pref_rows * expect_row_shape[0]):
                extra = np.zeros((expect_row_shape[0] * pref_rows - col.shape[0], expect_row_shape[1]))
                row = np.concatenate((col, extra), axis=0)
                cols_list[i] = row

        tiled_array = np.concatenate(cols_list, axis=1)

    else:
        tiled_array = np.concatenate(rows_list, axis=0)
    return tiled_array


def vis_slice(arrbot, arrtop, thresh=1e-4, opacity=0.5, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0,0,1,1])
        ax.set_xticks([])
        ax.set_yticks([])
    cmap = cm.viridis
    ntop = colors.Normalize()(arrtop)
    ctop = cmap(ntop)
    alphamap = np.ones_like(arrtop)*opacity
    alphamap[ntop<thresh] = 0.0
    ctop[...,-1] = alphamap

    axesims = []
    axesims.append(ax.imshow(arrbot, cmap='gray'))
    axesims.append(ax.imshow(ctop))
    return ax, axesims
