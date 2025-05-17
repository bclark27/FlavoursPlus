from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import sys
import os
import shutil
import subprocess

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

def generate_base_colors(average_rgb):
    """Generates 8 base colors from darkest to brightest based on an average RGB."""
    base_colors_rgb = []
    for i in np.linspace(0.2, 1.0, 8):  # Scale from 20% to 100% brightness
        scaled_rgb = tuple(int(c * i) for c in average_rgb)
        base_colors_rgb.append(scaled_rgb)
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
