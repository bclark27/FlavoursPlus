from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import sys
import os
import shutil
import subprocess
import colorsys
import numpy as np
from itertools import combinations

home_dir = '/home/ben/'

def color_similarity(color1, color2):
  """Calculates the Euclidean distance between two RGB colors (handles both tuples and arrays)."""
  color1 = np.array(color1)
  color2 = np.array(color2)
  return np.sqrt(np.sum((color1 - color2) ** 2))

def find_most_similar_subset(random_colors, target_colors):
  """
  Finds the subset of random colors that are most similar to the target colors.

  Args:
    random_colors: A list of RGB color tuples or NumPy arrays.
    target_colors: A list of RGB color tuples or NumPy arrays.

  Returns:
    A tuple containing:
      - A list of the 6 most similar random colors (in a matched order).
      - A list of the 2 least similar random colors.
  """
  if len(target_colors) != 6 or len(random_colors) != 8:
    raise ValueError("Target colors should be 6 and random colors should be 8.")

  best_similarity_score = float('inf')
  best_matching_subset = None
  remaining_colors = None

  # Convert random_colors to a list of tuples for use with combinations and sets
  random_colors_tuples = [tuple(color) for color in random_colors]

  for combo_tuple in combinations(random_colors_tuples, 6):
    combo = list(combo_tuple) # Convert back to list for easier use later
    current_similarity_score = 0
    remaining_tuples = list(set(random_colors_tuples) - set(combo_tuple))
    remaining = [list(color_tuple) for color_tuple in remaining_tuples] # Convert back to lists

    # Calculate the minimum distance for each target color to the current combo
    for target_color in target_colors:
      min_distance_to_combo = float('inf')
      for random_color in combo:
        distance = color_similarity(target_color, random_color)
        min_distance_to_combo = min(min_distance_to_combo, distance)
      current_similarity_score += min_distance_to_combo

    if current_similarity_score < best_similarity_score:
      best_similarity_score = current_similarity_score
      best_matching_subset = list(combo)
      remaining_colors = remaining

  # Now, we need to order the best matching subset to correspond to the target colors
  ordered_subset = []
  available_subset = list(best_matching_subset)

  for target_color in target_colors:
    best_match = None
    min_distance = float('inf')

    for color in available_subset:
      distance = color_similarity(target_color, color)
      if distance < min_distance:
        min_distance = distance
        best_match = color

    ordered_subset.append(best_match)
    available_subset.remove(best_match)

  return ordered_subset, remaining_colors

def hex_to_rgb(hex_color):
    """Converts a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Converts an RGB tuple to a hex color string."""
    return '#%02x%02x%02x' % rgb_color

def get_average_color(image_path):
    """Calculates the average color of an image."""
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img.getdata()).reshape(-1, 3)
    average = np.mean(pixels, axis=0).astype(int)
    return tuple(average)

def generate_shades(rgb, num_shades):
    """
    Generates a list of shades for a given hex color.

    Args:
        hex_color: The hex color string (e.g., "#RRGGBB").
        num_shades: The number of shades to generate.

    Returns:
        A list of hex color strings representing the shades.
    """
    
    rgb = tuple(x / 255 for x in rgb)

    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)

    # Generate shades by varying lightness
    shade_values = np.linspace(0.0, 1.0, num_shades)
    shades_hls = [(h, shade, s) for shade in shade_values]

    # Convert shades back to RGB and then to hex
    shades_rgb = [colorsys.hls_to_rgb(*hls) for hls in shades_hls]
    shades_255_rgb = [tuple((int(r * 255), int(g * 255), int(b * 255))) for r, g, b in shades_rgb]

    return shades_255_rgb

def generate_base_colors(average_rgb):
    """Generates 8 base colors with more defined dark, mid, and bright ranges."""
    shades = generate_shades(average_rgb, 26)
    base_colors_rgb = [
        shades[3],
        shades[6],
        shades[8],
        shades[10],
        shades[18],
        shades[20],
        shades[22],
        shades[24],
    ]
    return [rgb_to_hex(c) for c in base_colors_rgb]


def generate_accent_colors(image_path, num_colors=8):
    """Generates accent colors using K-means clustering on the image pixels."""
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img.getdata()).reshape(-1, 3)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10)
    kmeans.fit(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    
    clusters = [(x[0], x[1], x[2]) for x in cluster_centers]

    target6 = [
        hex_to_rgb('#a54242'),
        hex_to_rgb('#de935f'),
        hex_to_rgb('#f0c674'),
        hex_to_rgb('#b5bd68'),
        hex_to_rgb('#81a2be'),
        hex_to_rgb('#5f819d'),
    ]

    matches, remaining = find_most_similar_subset(cluster_centers, target6)

    f = 0.5
    for i in range(len(matches)):
        m = matches[i]
        n = [0,0,0]
        #n[0] = int((m[0] + target6[i][0]) / 2)
        n[0] = int((target6[i][0] - m[0]) * f + m[0])
        n[1] = int((target6[i][1] - m[1]) * f + m[1])
        n[2] = int((target6[i][2] - m[2]) * f + m[2])
        matches[i] = tuple(n)

    matches += remaining

    return [rgb_to_hex(tuple(c)) for c in matches]

def format_terminal_colors(base_colors, accent_colors, name):
    output = f"scheme: \"{name}\"\nauthor: \"me\"\n"

    base_colors += accent_colors
    for i, color in enumerate(base_colors):
        num = "{0:0{1}x}".format(i,2).upper()
        color = color.lstrip('#').upper()
        output += f"base{num}: \"{color}\"\n"

    return output

def set_nitrogen(image_path):
# Configure Nitrogen to use the provided image as wallpaper
    try:
        subprocess.run(['nitrogen', '--set-zoom-fill', image_path], check=True)
        print(f"\nSet '{image_path}' as the desktop background using Nitrogen.")
    except FileNotFoundError:
        print("\nWarning: Nitrogen command not found. Please ensure it is installed to set the wallpaper.")
    except subprocess.CalledProcessError as e:
        print(f"\nError setting wallpaper with Nitrogen: {e}")

def add_yaml_to_flavours(scheme_file, scheme_name):
    dest_dir = f"{home_dir}.local/share/flavours/base16/schemes/{scheme_name}"
    print(dest_dir)
    try:
        os.makedirs(dest_dir)
    except:
        pass

    dest_file = f'{dest_dir}/{scheme_name}.yaml'
    print(dest_file)

    if os.path.exists(dest_file):
        # in case of the src and dst are the same file
        if os.path.samefile(scheme_file, dest_file):
            return
        os.remove(dest_file)
    shutil.copyfile(scheme_file, dest_file)

def apply_flavours(scheme_name):
    subprocess.run(['flavours', 'apply', scheme_name], check=True)
    subprocess.run(['i3-msg', 'restart'], check=True)


def main():

    if len(sys.argv) != 3:
        print("Usage: python image_colors.py <image_path> <scheme_name>")
        sys.exit(1)

    image_path = sys.argv[1]
    scheme_name = sys.argv[2]
    output_file = f"{scheme_name}.yaml"

    average_color_rgb = get_average_color(image_path)
    base_colors = generate_base_colors(average_color_rgb)
    accent_colors = generate_accent_colors(image_path)
    terminal_colors = format_terminal_colors(base_colors, accent_colors, scheme_name)
    print("\nGenerated 16-color scheme for Linux terminal:\n")
    print(terminal_colors)


    try:
        with open(output_file, "w") as f:
            f.write(terminal_colors)

    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

    set_nitrogen(image_path)
    add_yaml_to_flavours(output_file, scheme_name)
    apply_flavours(scheme_name)

if __name__ == "__main__":
    main()
