import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
                & (img[:, :, 1] > rgb_thresh[1]) \
                & (img[:, :, 2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# The pixels for a rock appear to be both higher than 110 on the red and green channels
# and lower than 50 on the blue channel
def find_rocks(img, rgb_thresh=(110, 110, 50)):
    rockpix = ((img[:, :, 0] > rgb_thresh[0]) \
               & (img[:, :, 1] > rgb_thresh[1]) \
               & (img[:, :, 2] < rgb_thresh[2]))

    color_select = np.zeros_like(img[:, :, 0])
    color_select[rockpix] = 1
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from dead ahead for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180

    # perform polar equation.
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))  # keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), m, (img.shape[1], img.shape[0]))

    return warped, mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img
    # 1) Define source and destination points for perspective transform

    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])

    # 2) Apply perspective transform
    warped, mask = perspect_transform(image, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples

    threshed = color_thresh(warped)

    yellow_threshed = find_rocks(warped)

    # to simplify things, define the obstacles image as the inverse of the terrain.
    threshold_obstacles = np.absolute(np.float32(threshed) - 1) * mask

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    Rover.vision_image[:, :, 0] = threshold_obstacles * 255
    Rover.vision_image[:, :, 1] = yellow_threshed * 255
    Rover.vision_image[:, :, 2] = threshed * 255

    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    xpix_obs, ypix_obs = rover_coords(threshold_obstacles)

    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size

    # 6) Convert rover-centric pixel values to world coordinates
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, Rover.pos[0], 
                                                        Rover.pos[1], Rover.yaw, world_size, scale)

    obs_x_world, obs_y_world = pix_to_world(xpix_obs, ypix_obs, Rover.pos[0],
                                                         Rover.pos[1], Rover.yaw, world_size, scale)

    if yellow_threshed.any():
        xpix_rock, ypix_rock = rover_coords(yellow_threshed)
        rock_x_world, rock_y_world = pix_to_world(xpix_rock, ypix_rock, Rover.pos[0],
                                                        Rover.pos[1], Rover.yaw, world_size, scale)
        rock_distances, rock_angles = to_polar_coords(xpix_rock, ypix_rock)
        rock_idx = np.argmin(rock_distances)
        rock_x = rock_x_world[rock_idx]
        rock_y = rock_y_world[rock_idx]

        Rover.worldmap[rock_y, rock_x, 1] = 255
        Rover.vision_image[:,:,1] = yellow_threshed * 255
    else:
        Rover.vision_image[:, :, 1] = 0

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1

    # Since there is a chance that the indexes of these arrays may overlap, give the navigable
    # pixels more weight and let the create_output_images function decide.
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 10
    Rover.worldmap[obs_y_world, obs_x_world, 0] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    distances, angles = to_polar_coords(xpix, ypix)

    # Update Rover pixel distances and angles
    Rover.nav_dists = distances
    Rover.nav_angles = angles
    
    return Rover
