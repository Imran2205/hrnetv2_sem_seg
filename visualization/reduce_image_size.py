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
    return os.path.relpath(str(file_path), str(root_path))


def calculate_new_dimensions(width, height, scale_percent):
    """
    Calculate new dimensions based on scale percentage
    """
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    return new_width, new_height


def reduce_image_size(source_path, dest_path, quality_percent, scale_percent, target_dpi=None):
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with Image.open(source_path) as img:
            # Get original format, DPI, and dimensions
            img_format = img.format
            original_dpi = img.info.get('dpi', (72, 72))
            original_width, original_height = img.size

            new_width, new_height = calculate_new_dimensions(
                original_width, original_height, scale_percent)

            if scale_percent < 100:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            original_size = os.path.getsize(source_path) / 1024  # Size in KB

            dpi_to_use = (target_dpi, target_dpi) if target_dpi else original_dpi

            if img_format == 'PNG':
                if img.mode == 'RGBA':
                    img.save(dest_path, format='PNG', optimize=True,
                             dpi=dpi_to_use)
                else:
                    if img.mode != 'P':
                        try:
                            img_conv = img.convert('P', palette=Image.ADAPTIVE)
                            img_conv.save(dest_path, format='PNG', optimize=True,
                                          dpi=dpi_to_use)
                        except Exception:
                            img.save(dest_path, format='PNG', optimize=True,
                                     dpi=dpi_to_use)
                    else:
                        img.save(dest_path, format='PNG', optimize=True,
                                 dpi=dpi_to_use)
            else:
                img.save(dest_path, format=img_format, quality=quality_percent,
                         optimize=True, dpi=dpi_to_use)

            new_size = os.path.getsize(dest_path) / 1024  # Size in KB

            return True, original_size, new_size, original_dpi, dpi_to_use, \
                (original_width, original_height), (new_width, new_height)

    except Exception as e:
        return False, 0, 0, (0, 0), (0, 0), (0, 0), (0, 0), str(e)


def process_folder(source_folder, quality_percent, scale_percent, target_dpi=None):
    dest_folder = create_resized_directory(source_folder)

    IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    total_files = 0
    processed_images = 0
    copied_files = 0
    total_saved = 0
    failed_files = []

    start_time = time.time()
    source_folder = Path(source_folder)

    for source_path in source_folder.rglob('*'):
        if source_path.is_file():
            total_files += 1
            rel_path = get_relative_path(source_path, source_folder)
            dest_path = os.path.join(dest_folder, rel_path)

            if source_path.suffix in IMAGE_FORMATS:
                success, original_size, new_size, original_dpi, new_dpi, \
                    original_dims, new_dims = reduce_image_size(
                    str(source_path), dest_path, quality_percent,
                    scale_percent, target_dpi)

                if success:
                    processed_images += 1
                    saved_size = original_size - new_size
                    total_saved += saved_size
                    print(f"\nProcessed Image: {rel_path}")
                    print(f"Original size: {original_size:.2f}KB")
                    print(f"New size: {new_size:.2f}KB")
                    print(f"Saved: {saved_size:.2f}KB")
                    print(f"Original dimensions: {original_dims[0]}x{original_dims[1]}")
                    print(f"New dimensions: {new_dims[0]}x{new_dims[1]}")
                    print(f"Original DPI: {original_dpi}")
                    print(f"New DPI: {new_dpi}")
                else:
                    failed_files.append(rel_path)
            else:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)
                copied_files += 1
                print(f"\nCopied file: {rel_path}")

    end_time = time.time()

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
    """
    This script will save resized images in source_folder_name_resized directory.
    """
    folder_path = "/Users/ibk5106/Desktop/research/under_water/hrnetv2_sem_seg/visualization/figures"
    quality_percent = 50
    scale_percent = 50
    target_dpi = 700

    if not 1 <= quality_percent <= 100:
        print("Quality percentage must be between 1 and 100")
        return

    if not 1 <= scale_percent <= 100:
        print("Scale percentage must be between 1 and 100")
        return

    if target_dpi is not None and target_dpi <= 0:
        print("DPI must be a positive number")
        return

    if not os.path.exists(folder_path):
        print("Source folder does not exist")
        return

    process_folder(folder_path, quality_percent, scale_percent, target_dpi)


if __name__ == "__main__":
    main()