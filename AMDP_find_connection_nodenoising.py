# Overall Function: This code processes chiral spectral data in SPC format to analyze and visualize polarization correlations.
# It calculates dissymmetry factors and correlation functions, identifies peaks and valleys using the AMPD algorithm,
# visualizes results in 3D and 2D plots with annotated extrema, and connects extreme points across different spectral curves.
# The primary goal is to explore the relationship between spectral characteristics and rotation angles through quantitative analysis and visualization.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyspectra.readers.read_spc import read_spc
from pyspectra.readers.read_spc import read_spc_dir
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from scipy.interpolate import interp1d


# Calculate the normalized dissymmetry factor
# Parameters:
#   s1, s2: Spectral data from two different polarization states
#   lamp1, lamp2: Reference lamp spectra for normalization
# Returns: Dissymmetry factor array normalized by lamp spectra
def dissymmetry_factor(s1, s2, lamp1, lamp2):
    return 2 * np.multiply(np.multiply(s1, np.reciprocal(lamp1)) - np.multiply(s2, np.reciprocal(lamp2)),
                           np.reciprocal(np.multiply(s1, np.reciprocal(lamp1)) + np.multiply(s2, np.reciprocal(lamp2))))


# Calculate the unnormalized dissymmetry factor (without lamp spectra correction)
# Parameters:
#   s1, s2: Spectral data from two different polarization states
# Returns: Unnormalized dissymmetry factor array
def dissymmetry_factor_unnorm(s1, s2):
    return 2 * np.multiply((s1 - s2), np.reciprocal(s1 + s2))


# Adjust the lightness of a given color
# Parameters:
#   color: Input color (name or RGB/RGBA value)
#   amount: Scaling factor for lightness (e.g., >1 increases lightness, <1 decreases it)
# Returns: Adjusted color in RGB format
def adjust_lightness(color, amount):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# Below we define the correlation functions:
# correlation(x,y) - unnormalized cross-correlation function
# correlation_2(x,y) - cross-correlation function normalized by the autocorrelations of x and y

# Compute unnormalized cross-correlation between two arrays
# Parameters:
#   x, y: Input arrays to correlate
# Returns: Full-mode cross-correlation result
def correlation(x, y):
    result = np.correlate(x, y, mode='full')
    return result


# Compute normalized cross-correlation (normalized by square root of autocorrelation products)
# Parameters:
#   x, y: Input arrays to correlate
# Returns: Normalized full-mode cross-correlation result
def correlation_2(x, y):
    return np.divide(np.correlate(x, y, mode='full'),
                     np.sqrt(np.multiply(np.correlate(x, x, mode='full'), np.correlate(y, y, mode='full'))))


# Auxiliary functions to find the value in an array "nums" that is closest to "value"
# Parameters:
#   nums: Input array to search
#   value: Target value to find the closest match to
#   position: Strategy for multiple matches (0: middle of range, 1: lower middle)
# Returns: Closest value and its index in "nums"
def find_closest_value(nums, value, position):
    diff = np.abs(nums - np.ones_like(nums) * value)
    min_diff = min(diff)
    closest_idx = np.array(np.where(diff == min_diff)[0])
    if len(closest_idx) <= 1:
        return nums[closest_idx], closest_idx
    else:
        if position == 0:
            middle_idx = np.ceil(len(closest_idx) / 2)
        elif position == 1:
            middle_idx = np.floor(len(closest_idx) / 2)
        return nums[closest_idx[int(middle_idx)]], closest_idx[int(middle_idx)]


# Find closest values to a target on both sides of the global maximum in an array
# Parameters:
#   nums: Input array
#   value: Target value to find matches for
# Returns: Closest values and their indices (left and right of the global maximum)
def find_closest_single_extr(nums, value):
    idx = np.array([k for k in range(len(nums))])
    before_max = np.array(np.where(idx < np.argmax(nums))[0])
    closest_left, left_idx = find_closest_value(np.array([nums[k] for k in before_max]), value, 0)
    after_max = np.array(np.where(idx >= np.argmax(nums))[0])
    closest_right, right_idx = find_closest_value(np.array([nums[k] for k in after_max]), value, 1)
    return np.vstack((np.append(closest_left, closest_right), np.append(left_idx, len(before_max) + right_idx)))


# Creating gradient filling for the area under a 2D plot in 3D space

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Generate a custom linear colormap from start_color to end_color
# Parameters:
#   start_color, end_color: RGB/RGBA colors defining the gradient
# Returns: LinearSegmentedColormap object
def color_grad(start_color, end_color):
    start_rgba = plt.matplotlib.colors.to_rgba(start_color, alpha=1)
    end_rgba = plt.matplotlib.colors.to_rgba(end_color, alpha=1)
    return LinearSegmentedColormap.from_list("custom", [start_rgba, end_rgba])


# Fill the area between two curves in 3D with a color gradient
# Parameters:
#   ax: 3D axes object to plot on
#   width_line: Reference array for gradient normalization
#   carrier: X-axis data for the curves
#   curve1, curve2: Y-axis data defining the upper and lower bounds of the filled area
#   y_value: Fixed Y-position in 3D space for the filled area
#   start_color, end_color: Colors for the gradient
def gradient_fill(ax, width_line, carrier, curve1, curve2, y_value, start_color, end_color):
    cmap_left = color_grad(end_color, start_color)
    cmap_right = color_grad(start_color, end_color)
    cmap_middle = color_grad(start_color, start_color)
    norm_left = plt.Normalize(min(carrier), width_line[int(0.25 * len(width_line))])
    norm_right = plt.Normalize(width_line[int(0.75 * len(width_line))], max(carrier))
    norm_middle = plt.Normalize(width_line[int(0.25 * len(width_line))], width_line[int(0.75 * len(width_line))])

    polygons = []
    colors = []

    for i in range(len(carrier) - 1):
        verts = [(carrier[i], y_value, curve1[i]), (carrier[i + 1], y_value, curve1[i + 1]),
                 (carrier[i + 1], y_value, curve2[i + 1]), (carrier[i], y_value, curve2[i])]
        polygons.append(verts)
        mid_x = (carrier[i] + carrier[i + 1]) / 2
        if carrier[i] <= width_line[int(0.25 * len(width_line))]:
            colors.append(cmap_left(norm_left(mid_x)))
        elif carrier[i] >= width_line[int(0.75 * len(width_line))]:
            colors.append(cmap_right(norm_right(mid_x)))
        else:
            colors.append(cmap_middle(norm_middle(mid_x)))

    poly = Poly3DCollection(polygons, facecolors=colors, edgecolors='none')
    poly.set_alpha(1)
    ax.add_collection3d(poly)

    # Check concavity of the area contour
    def concave_check(input, position):
        lda = 1 / 2
        res_check = []
        n = 0
        for i in range(len(input) - 2):
            for j in range(i + 2, len(input)):
                res_check = np.append(res_check, bool(
                    find_closest_value(input, input[i] * lda + input[j] * (1 - lda), position)[0] >=
                    find_closest_value(input, input[i], position)[0] * lda +
                    find_closest_value(input, input[j], position)[0] * (1 - lda)))
                n += 1
        return len(np.where(res_check == True)[0]) / n

    # Calculate the area under a curve using the trapezoidal rule
    def calc_area(carrier, input):
        return np.sum(np.multiply(np.diff(carrier), input[:-1] + input[1:]) / 2)

    # Calculate dissymmetry of the area around the global maximum
    def dyssim_area(carrier, before_max, after_max, input):
        reduced_area_before_max = calc_area(carrier[before_max:np.argmax(input)],
                                            input[before_max:np.argmax(input)]) * concave_check(
            input[before_max:np.argmax(input):int(len(carrier) / 200)], 0)
        reduced_area_after_max = calc_area(carrier[np.argmax(input):after_max],
                                           input[np.argmax(input):after_max]) * concave_check(
            input[np.argmax(input):after_max:int(len(carrier) / 200)], 1)
        return np.round_((reduced_area_after_max - reduced_area_before_max) / (
                reduced_area_after_max + reduced_area_before_max) * 100, decimals=1)

# Implementation of the AMPD algorithm for peak detection (finds local maxima)
# Parameters:
#   data: 1D numpy array to analyze
# Returns: Indices of local maxima in "data"
def AMPD(data):
    """
    Implements the AMPD (Automatic Multiscale Peak Detection) algorithm
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]


# AMPD-based algorithm to find local minima (by inverting the signal)
# Parameters:
#   data: 1D numpy array to analyze
# Returns: Indices of local minima in "data"
def AMPD_min(data):
    """
    Finds local minima by inverting the signal (converting minima to maxima) and applying AMPD
    """
    inverted_data = -data  # Invert signal to turn minima into maxima
    max_indices = AMPD(inverted_data)  # Detect maxima in inverted signal
    return max_indices  # These indices correspond to minima in the original signal


# Example usage: Analyze global maxima of each spectrum and highlight with dashed lines

# Read reference lamp spectra
lamp1 = read_spc(
    'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_01.spc')
lamp2 = read_spc(
    'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_02.spc')

# Mapping from rotation angles to spectrum IDs (right and left helices)
r_ID = {0: "28", 15: "26", 30: "24", 45: "22", 60: "20", 75: "18", 90: "16",
        105: "14", 120: "12", 135: "10", 150: "09", 165: "06", 180: "04"}
l_ID = {0: "29", 15: "27", 30: "25", 45: "23", 60: "21", 75: "19", 90: "17",
        105: "15", 120: "13", 135: "11", 150: "08", 165: "07", 180: "05"}

# Generate discrete colors using HSV colormap
discr_colors = cm.hsv(np.delete(np.linspace(0, 1, len(list(r_ID.keys())) + 1), -1))

# Create 3D figure
fig = plt.figure(figsize=(8, 9))

# Initialize arrays to store correlation and dissymmetry limits
corr_limits = np.empty(len(r_ID))  # Ensure length matches data
g_limits = np.empty(len(r_ID))

# Add 3D subplot and set aspect ratio
ax = fig.add_subplot(projection='3d')
x_scaling = 2.0
y_scaling = 4.0
z_scaling = 2.0
ax.set_box_aspect([x_scaling, y_scaling, z_scaling])

# Store shifts of correlation maxima
shifts = []


# Formatter for scientific notation
def format_scientific(x, pos):
    return "{:.1e}".format(x)


sci_format = mticker.FuncFormatter(format_scientific)

# Store global min/max of correlation values for each curve
bottom_z_shifts = []
top_z_shifts = []

# First pass: Calculate global min/max of correlation functions for scaling
for i in range(len(list(r_ID.keys()))):
    s1 = read_spc(
        'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_' +
        list(l_ID.values())[i] + '.spc')
    s2 = read_spc(
        'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_' +
        list(l_ID.values())[i] + '.spc')

    g = dissymmetry_factor(s1, s2, lamp1, lamp2)
    corr_g = correlation_2(s1, s2)

# Second pass: Refine global min/max using right/left helix data
bottom_z_shifts = []
top_z_shifts = []

for i in range(len(r_ID)):
    s1 = read_spc(
        'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_' +
        list(r_ID.values())[i] + '.spc')
    s2 = read_spc(
        'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_' +
        list(l_ID.values())[i] + '.spc')

    g = dissymmetry_factor(s1, s2, lamp1, lamp2)
    corr_g = correlation_2(s1, s2)

    # Record global min and max for current correlation function
    bottom_z_shifts.append(min(corr_g))
    top_z_shifts.append(max(corr_g))

# Determine global z-axis limits
bottom_z_lim = min(bottom_z_shifts)
top_z_lim = max(top_z_shifts)

# Store all detected peaks, valleys, and connected extreme points
all_peaks = []
all_valleys = []
connected_extremes_x = []
connected_extremes_y = []
connected_extremes_z = []

# Process each rotation angle (in reverse order) to plot and analyze
for i in range(len(list(r_ID.keys())) - 1, -1, -1):
    # Read spectral data for current rotation angle (right and left helices)
    s1 = read_spc(
        'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_' +
        list(r_ID.values())[i] + '.spc')
    s2 = read_spc(
        'E:/chiral/Analysis-of-chiral-scattering-main/Analysis-of-chiral-scattering-main/Chiral data/2024_10_23_Zmaga_Sun/2024_10_23/Spectrum_(LS6)-2024_10_23-ID_' +
        list(l_ID.values())[i] + '.spc')

    # Calculate dissymmetry factor and normalized correlation
    g = dissymmetry_factor(s1, s2, lamp1, lamp2)
    corr_g = correlation_2(s1, s2)
    corr_carrier = np.array([2 * k / len(corr_g) - 1 for k in range(len(corr_g))])  # Normalized x-axis for correlation

    # Update limits
    corr_limits[i] = max(abs(np.array(corr_g)))
    g_limits = max(np.abs(np.array(g)))

    # Get position of maximum correlation
    shift = corr_carrier[np.argmax(corr_g)]

    # Plot dashed line through correlation maximum (spanning global z-range)
    ax.plot(
        np.ones(100) * corr_carrier[np.argmax(corr_g)],
        np.full(100, int(list(r_ID.keys())[i])),
        np.linspace(bottom_z_lim, top_z_lim, 100),
        color='white',
        linestyle='dashed',
        linewidth=3.0,
        path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()]
    )
    # Plot correlation function curve
    line = ax.plot(
        corr_carrier,
        np.full(len(corr_carrier), int(list(r_ID.keys())[i])),  # Fixed y (rotation angle) for current curve
        corr_g,
        color=adjust_lightness(discr_colors[i], 1.0),
        label='Max. pos. = {}'.format(np.round(shift, decimals=3)),
        linewidth=3.0,
        path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()]
    )

    # Detect peaks (local maxima) and valleys (local minima) using AMPD
    peaks = AMPD(corr_g)
    valleys = AMPD_min(corr_g)

    # Store detected extrema
    all_peaks.append(corr_g[peaks])
    all_valleys.append(corr_g[valleys])

    # Annotate peaks with red markers and values
    for peak in peaks:
        ax.scatter(corr_carrier[peak], int(list(r_ID.keys())[i]), corr_g[peak], color='red', s=50)
        ax.text(corr_carrier[peak], int(list(r_ID.keys())[i]), corr_g[peak], f'{corr_g[peak]:.2f}',
                color='red', fontsize=8, ha='center', va='bottom')

    # Annotate valleys with blue markers and values
    for valley in valleys:
        ax.scatter(corr_carrier[valley], int(list(r_ID.keys())[i]), corr_g[valley], color='blue', s=50)
        ax.text(corr_carrier[valley], int(list(r_ID.keys())[i]), corr_g[valley], f'{corr_g[valley]:.2f}',
                color='blue', fontsize=8, ha='center', va='top')

    # Record shift of correlation maximum
    shifts = np.append(shifts, shift)

    # Combine all extrema (peaks + valleys) for current curve
    all_extremes_x = np.concatenate([corr_carrier[peaks], corr_carrier[valleys]])
    all_extremes_z = np.concatenate([corr_g[peaks], corr_g[valleys]])

    # Set initial point for connecting extremes (first curve processed)
    if i == len(list(r_ID.keys())) - 1:
        # For the first curve, select the first extreme point as the starting point
        if len(all_extremes_x) > 0:
            # Modify the index here to change the starting extreme point
            connected_extremes_x.append(all_extremes_x[1])  # Current starting index: 1
            connected_extremes_y.append(int(list(r_ID.keys())[i]))
            connected_extremes_z.append(all_extremes_z[0])  # Matches x index for z-value
    else:
        # For subsequent curves, find the extreme closest to the previous curve's last extreme
        prev_x = connected_extremes_x[-1]
        distances = np.abs(all_extremes_x - prev_x)
        closest_index = np.argmin(distances)
        connected_extremes_x.append(all_extremes_x[closest_index])
        connected_extremes_y.append(int(list(r_ID.keys())[i]))
        connected_extremes_z.append(all_extremes_z[closest_index])

# Interpolate and plot the connected extreme points
if len(connected_extremes_x) > 1:
    connected_extremes_x = np.array(connected_extremes_x)
    connected_extremes_y = np.array(connected_extremes_y)
    connected_extremes_z = np.array(connected_extremes_z)

    # Sort by rotation angle (y-axis) for smooth interpolation
    sorted_indices = np.argsort(connected_extremes_y)
    connected_extremes_x = connected_extremes_x[sorted_indices]
    connected_extremes_y = connected_extremes_y[sorted_indices]
    connected_extremes_z = connected_extremes_z[sorted_indices]

    # Cubic interpolation for smooth curve
    f_x = interp1d(connected_extremes_y, connected_extremes_x, kind='cubic')
    f_z = interp1d(connected_extremes_y, connected_extremes_z, kind='cubic')

    # Generate fine y-axis points for interpolation
    y_fine = np.linspace(min(connected_extremes_y), max(connected_extremes_y), 100)
    x_fine = f_x(y_fine)
    z_fine = f_z(y_fine)

    # Plot the connected extremes curve
    ax.plot(x_fine, y_fine, z_fine, color='green', linewidth=2, label='Connected extremes')

# Configure 3D plot labels and settings
ax.set_xlabel(f'Normalized spectral lag, a.u.', labelpad=2)
ax.set_ylabel(f'Rotation angle, deg.', labelpad=10)
ax.legend(ncol=2, loc='upper right', fontsize=10)
ax.set_xticks([k / 2 for k in range(-2, 3, 1)])
ax.set_yticks(list(r_ID.keys()))
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-15, 195)
ax.set_zlim(bottom_z_lim, top_z_lim)
plt.title(f'Correlation of polarizations (Si substrate: C2 cell, left helix)')

# Make 3D panes transparent
ax.zaxis.set_pane_color((1, 1, 1, 0))
ax.yaxis.set_pane_color((1, 1, 1, 0))
ax.xaxis.set_pane_color((1, 1, 1, 0))

# Set 3D viewing angle
ax.view_init(30, -130, 0)

# Create a separate 2D plot for the connected extremes curve
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(y_fine, z_fine, color='green', linewidth=2, label='Connected extremes')
ax2.set_xlabel(f'Rotation angle, deg.')
ax2.set_ylabel('Correlation value')
ax2.legend()
ax2.set_title('Curve of connected extremes')

# Display plots
plt.show()

# Print all detected peaks and valleys
print("All peaks:", all_peaks)
print("All valleys:", all_valleys)