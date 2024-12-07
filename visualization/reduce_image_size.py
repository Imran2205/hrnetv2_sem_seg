from PIL import Image
import os
from pathlib import Path
import shutil
import time


def create_resized_directory(source_path):
    """
    Create a new directory with '_resized' suffix parallel to the source directory
    """
    source_path = Path(source_path)
    new_dir_name = f"{source_path.name}_resized"
    new_dir_path = source_path.parent / new_dir_name
    new_dir_path.mkdir(exist_ok=True)
    return str(new_dir_path)


def get_relative_path(file_path, root_path):
    """
    Get the relative path of a file with respect to the root path
    """
    return os.path.relpath(str(file_path), str(root_path))


def reduce_image_size(source_path, dest_path, quality_percent, target_dpi=None):
    """
    Reduce the file size of an image while maintaining resolution and format

    Args:
        source_path (str): Path to the source image file
        dest_path (str): Path where the processed image will be saved
        quality_percent (int): Desired quality percentage (1-100)
        target_dpi (int, optional): Desired DPI value. If None, original DPI is maintained
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Open the image
        with Image.open(source_path) as img:
            # Get original format and DPI
            img_format = img.format
            original_dpi = img.info.get('dpi', (72, 72))

            # Get original file size
            original_size = os.path.getsize(source_path) / 1024  # Size in KB

            # Set DPI if specified
            dpi_to_use = (target_dpi, target_dpi) if target_dpi else original_dpi

            if img_format == 'PNG':
                # For PNG, use optimize and reduce the number of colors if possible
                if img.mode == 'RGBA':
                    # Preserve alpha channel for PNGs with transparency
                    img.save(dest_path, format='PNG', optimize=True,
                             dpi=dpi_to_use)
                else:
                    # For non-transparent PNGs, we can try additional optimization
                    if img.mode != 'P':  # If not already using palette
                        # Convert to palette mode if image has less than 256 colors
                        try:
                            img_conv = img.convert('P', palette=Image.ADAPTIVE)
                            img_conv.save(dest_path, format='PNG', optimize=True,
                                          dpi=dpi_to_use)
                        except Exception:
                            # If palette conversion fails, save with basic optimization
                            img.save(dest_path, format='PNG', optimize=True,
                                     dpi=dpi_to_use)
                    else:
                        # Already in palette mode, just optimize
                        img.save(dest_path, format='PNG', optimize=True,
                                 dpi=dpi_to_use)
            else:
                # For JPEG and other formats, use quality parameter
                img.save(dest_path, format=img_format, quality=quality_percent,
                         optimize=True, dpi=dpi_to_use)

            # Get new file size
            new_size = os.path.getsize(dest_path) / 1024  # Size in KB

            return True, original_size, new_size, original_dpi, dpi_to_use

    except Exception as e:
        return False, 0, 0, (0, 0), (0, 0), str(e)


def process_folder(source_folder, quality_percent, target_dpi=None):
    """
    Process all images in a folder and its subfolders
    """
    # Create resized directory
    dest_folder = create_resized_directory(source_folder)

    # Supported image formats
    IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # Statistics
    total_files = 0
    processed_images = 0
    copied_files = 0
    total_saved = 0
    failed_files = []

    start_time = time.time()
    source_folder = Path(source_folder)

    # Walk through all files in the folder and subfolders
    for source_path in source_folder.rglob('*'):
        if source_path.is_file():
            total_files += 1
            # Get the relative path to maintain directory structure
            rel_path = get_relative_path(source_path, source_folder)
            dest_path = os.path.join(dest_folder, rel_path)

            if source_path.suffix in IMAGE_FORMATS:
                success, original_size, new_size, original_dpi, new_dpi = reduce_image_size(
                    str(source_path), dest_path, quality_percent, target_dpi)

                if success:
                    processed_images += 1
                    saved_size = original_size - new_size
                    total_saved += saved_size
                    print(f"\nProcessed Image: {rel_path}")
                    print(f"Original size: {original_size:.2f}KB")
                    print(f"New size: {new_size:.2f}KB")
                    print(f"Saved: {saved_size:.2f}KB")
                    print(f"Original DPI: {original_dpi}")
                    print(f"New DPI: {new_dpi}")
                else:
                    failed_files.append(rel_path)
            else:
                # Copy non-image files
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)
                copied_files += 1
                print(f"\nCopied file: {rel_path}")

    end_time = time.time()

    # Print summary
    print("\nSummary:")
    print(f"Total files processed: {total_files}")
    print(f"Images processed: {processed_images}")
    print(f"Other files copied: {copied_files}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Total space saved: {total_saved / 1024:.2f}MB")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Output directory: {dest_folder}")

    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"- {file}")


def main():
    # Get input from user
    folder_path = "/Users/ibk5106/Desktop/research/under_water/hrnetv2_sem_seg/visualization/figures"
    quality_percent = 50
    dpi_input = 900

    # Convert DPI input
    target_dpi = int(dpi_input) if dpi_input else None

    # Validate quality percentage
    if not 1 <= quality_percent <= 100:
        print("Quality percentage must be between 1 and 100")
        return

    # Validate DPI if provided
    if target_dpi is not None and target_dpi <= 0:
        print("DPI must be a positive number")
        return

    # Check if folder exists
    if not os.path.exists(folder_path):
        print("Source folder does not exist")
        return

    # Process the folder
    process_folder(folder_path, quality_percent, target_dpi)


if __name__ == "__main__":
    main()