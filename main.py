from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import sys
import os
import shutil
import subprocess
import colorsys

home_dir = '/home/ben/'

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
    shades = generate_shades(average_rgb, 16)
    base_colors_rgb = [
        shades[2],
        shades[3],
        shades[5],
        shades[7],
        shades[9],
        shades[11],
        shades[13],
        shades[14],
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
    return [rgb_to_hex(tuple(c)) for c in cluster_centers]

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
