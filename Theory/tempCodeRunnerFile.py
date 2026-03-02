import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

def calc_theta(h, D):
    """
    Calculates the internal angle (theta) based on the dimensionless liquid height.
    h_ratio: h = H/D (Liquid height H divided by pipe diameter D).
    Returns theta in radians.
    """
    # Formula: theta = 2 * arccos(1 - 2h)
    # Constrain input to valid [-1, 1] range for arccos to avoid NaN errors due to float precision
    h_ratio = h/D
    val = np.clip(1.0 - 2.0 * h_ratio, -1.0, 1.0)
    theta = 2.0 * np.arccos(val)
    return theta

def calc_fractions(h, D):
    """
    Calculates the theoretical liquid volume fraction (alpha_L) from the internal angle.
    Returns a value between 0.0 and 1.0.
    """
    # Formula: alpha_L = (theta - sin(theta)) / (2 * pi)
    theta = calc_theta(h, D)
    alpha_L = (theta - np.sin(theta)) / (2.0 * np.pi)
    alpha_I = 1.0 - alpha_L
    return alpha_L, alpha_I

def calc_perimeters(h, D):
    """
    Calculates the wetted perimeter (s_L) and interface perimeter (s_I).
    D: Pipe diameter in mm (or any consistent unit).
    Returns (s_L, s_I) in the same units as D.
    """
    # Formula: s_L = (theta * D) / 2, s_I = D * sin(theta / 2)
    theta = calc_theta(h, D)
    s_L = (theta * D) / 2.0
    s_I = D * np.sin(theta / 2.0)
    return s_L, s_I


def compute_liquid_height(mask):
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return 0
    return ys.max() - ys.min()

def compute_local_heights(mask):
    """
    Calculates the height of the white region for every column (x-coordinate).
    Returns an array of heights in pixels.
    """
    heights = np.zeros(mask.shape[1])
    
    for x in range(mask.shape[1]):
        # Find white pixels in the current column
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            # +1 ensures a single pixel counts as height 1
            heights[x] = ys.max() - ys.min() + 1 
            
    return heights

# --- MAIN PART ---

# 1. Load the image as grayscale
img_path = r'C:\Users\Mateusz\Desktop\CODE\Heat-Pipes-3d-Modelling\Theory\Figure_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"OpenCV could not find or read the image at: '{img_path}'. Please check the path and filename.")

# 2. Extract ROI (Region of Interest)
# You need to crop out the text. Looking at the image, the pipe is roughly in the middle.
# These coordinates are estimates; adjust them based on your actual image dimensions.
pipe_top = 450  
pipe_bottom = 550 
liquid_mask = img[pipe_top:pipe_bottom, :]

# 3. Apply a threshold to ensure it's strictly binary (0 or 255)
_, liquid_mask = cv2.threshold(liquid_mask, 127, 255, cv2.THRESH_BINARY)

# 4. Physical parameters
D = 20.0  # pipe diameter in mm
mm_per_pixel = 0.1  # conversion factor

# 5. Calculate arrays of local properties
H_pixels_array = compute_local_heights(liquid_mask)
H_mm_array = np.clip(H_pixels_array * mm_per_pixel, 0.0, D)

# Calculate properties for every x-coordinate
alpha_L_array, alpha_G_array = calc_fractions(H_mm_array, D)
S_L_array, S_I_array = calc_perimeters(H_mm_array, D)

# Calculate averages for the whole pipe section
avg_liquid_fraction = np.mean(alpha_L_array)
avg_liquid_height = np.mean(H_mm_array)

def plot_3d_reconstruction(H_mm_array, D, mm_per_pixel):
    """
    Reconstructs and plots the 3D liquid interface inside a cylindrical pipe.
    """
    R = D / 2.0
    
    # 1. Create X coordinates (length of the pipe)
    x_pixels = np.arange(len(H_mm_array))
    X = x_pixels * mm_per_pixel
    
    # 2. Calculate Z coordinates (height of the interface)
    # H_mm is measured from the bottom, so we shift it down by R
    Z_interface = -R + H_mm_array
    
    # 3. Calculate Y coordinates (depth/width of the interface chord)
    # np.maximum prevents negative values inside sqrt due to minor float precision issues
    Y_max = np.sqrt(np.maximum(R**2 - Z_interface**2, 0))
    
    # 4. Create a 2D meshgrid to draw the 3D surface
    # 'v' is a normalized parameter to stretch across the width from -Y to +Y
    V = np.linspace(-1, 1, 50) 
    X_grid, V_grid = np.meshgrid(X, V)
    
    Z_grid = np.tile(Z_interface, (50, 1))
    Y_grid = V_grid * np.tile(Y_max, (50, 1))
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the liquid-gas interface (Blue Surface)
    ax.plot_surface(X_grid, Y_grid, Z_grid, color='cyan', alpha=0.7, antialiased=True)
    
    # Plot the bottom half of the pipe wall for reference (Gray Surface)
    # Theta goes from pi to 2*pi (the bottom half of a circle)
    theta = np.linspace(np.pi, 2*np.pi, 50)
    X_pipe, Theta_pipe = np.meshgrid(X, theta)
    Y_pipe = R * np.cos(Theta_pipe)
    Z_pipe = R * np.sin(Theta_pipe)
    ax.plot_surface(X_pipe, Y_pipe, Z_pipe, color='gray', alpha=0.2)
    
    # Formatting
    ax.set_xlabel('Pipe Length X (mm)')
    ax.set_ylabel('Depth Y (mm)')
    ax.set_zlabel('Height Z (mm)')
    ax.set_title('3D Reconstruction of Stratified/Slug Flow')
    
    # Force the axes to have proportional aspect ratios so the pipe doesn't look stretched
    # We set X to its actual length, and Y/Z to the pipe diameter
    ax.set_box_aspect([np.ptp(X), D, D]) 
    
    plt.show()

# --- Call this right at the end of your MAIN PART ---
# (Make sure to run this AFTER you have calculated H_mm_array)
plot_3d_reconstruction(H_mm_array, D, mm_per_pixel)