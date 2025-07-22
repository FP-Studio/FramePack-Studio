import os
import sys
import requests
import tarfile
import zipfile
import shutil
from tqdm import tqdm


def setup_ffmpeg():
    """Download and set up a cross-platform, full build of FFmpeg and FFprobe."""
    # Get the directory of the current script, which is now inside 'modules/toolbox/'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The 'bin' directory is created directly inside this script's directory.
    bin_dir = os.path.join(script_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    # --- Platform-specific configuration ---
    if sys.platform == "win32":
        platform = "windows"
        ffmpeg_name = "ffmpeg.exe"
        ffprobe_name = "ffprobe.exe"
        download_url = "https://github.com/GyanD/codexffmpeg/releases/download/7.0/ffmpeg-7.0-full_build.zip"
        archive_name = "ffmpeg.zip"
        # For Windows, the path is static and predictable
        path_in_archive_to_bin = "ffmpeg-7.0-full_build/bin"
    elif sys.platform.startswith("linux"):
        platform = "linux"
        ffmpeg_name = "ffmpeg"
        ffprobe_name = "ffprobe"
        # This link always points to the latest static build
        download_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        archive_name = "ffmpeg.tar.xz"
        # --- CHANGE: We no longer hardcode the path_in_archive_to_bin for Linux ---
    else:
        print(f"Unsupported platform: {sys.platform}")
        print(
            "Please download FFmpeg manually and place ffmpeg/ffprobe in the 'bin' directory."
        )
        return

    ffmpeg_path = os.path.join(bin_dir, ffmpeg_name)
    ffprobe_path = os.path.join(bin_dir, ffprobe_name)

    if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
        print(f"FFmpeg is already set up in: {bin_dir}")
        return

    archive_path = os.path.join(bin_dir, archive_name)

    try:
        print(f"FFmpeg not found. Downloading and setting up for {platform}...")
        download_ffmpeg(download_url, archive_path)

        print("Download complete. Installing...")
        temp_extract_dir = os.path.join(bin_dir, "temp_ffmpeg_extract")
        os.makedirs(temp_extract_dir, exist_ok=True)

        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(path=temp_extract_dir)
        elif archive_name.endswith(".tar.xz"):
            with tarfile.open(archive_path, "r:xz") as archive:
                archive.extractall(path=temp_extract_dir)

        # --- ROBUSTNESS CHANGE FOR LINUX ---
        # Dynamically find the path to the binaries instead of hardcoding it.
        if platform == "linux":
            # Find the single subdirectory inside the extraction folder
            subdirs = [
                d
                for d in os.listdir(temp_extract_dir)
                if os.path.isdir(os.path.join(temp_extract_dir, d))
            ]
            if len(subdirs) != 1:
                raise Exception(
                    f"Expected one subdirectory in Linux FFmpeg archive, but found {len(subdirs)}."
                )
            # The binaries are directly inside this discovered folder
            source_bin_dir = os.path.join(temp_extract_dir, subdirs[0])
        else:  # For Windows, we use the predefined path
            source_bin_dir = os.path.join(temp_extract_dir, path_in_archive_to_bin)

        # Find the executables in the now correctly identified source folder and copy them
        source_ffmpeg_path = os.path.join(source_bin_dir, ffmpeg_name)
        source_ffprobe_path = os.path.join(source_bin_dir, ffprobe_name)

        if not os.path.exists(source_ffmpeg_path) or not os.path.exists(
            source_ffprobe_path
        ):
            raise FileNotFoundError(
                f"Could not find ffmpeg/ffprobe in the expected location: {source_bin_dir}"
            )

        shutil.copy(source_ffmpeg_path, ffmpeg_path)
        shutil.copy(source_ffprobe_path, ffprobe_path)

        if platform == "linux":
            os.chmod(ffmpeg_path, 0o755)
            os.chmod(ffprobe_path, 0o755)

        print(f"✅ FFmpeg setup complete. Binaries are in: {bin_dir}")

    except Exception as e:
        print(f"\n❌ Error setting up FFmpeg: {e}")
        import traceback

        traceback.print_exc()
        print(
            "\nPlease download FFmpeg manually and place the 'ffmpeg' and 'ffprobe' executables in the 'bin' directory."
        )
        print("Download for Windows: https://www.gyan.dev/ffmpeg/builds/")
        print("Download for Linux: https://johnvansickle.com/ffmpeg/")
    finally:
        # Clean up
        if os.path.exists(archive_path):
            os.remove(archive_path)
        if "temp_extract_dir" in locals() and os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)


def download_ffmpeg(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    # The calling function now handles the initial "Downloading..." message.
    # This keeps the download function focused on its single responsibility.
    with (
        open(destination, "wb") as file,
        tqdm(
            desc=os.path.basename(
                destination
            ),  # Use basename for a cleaner progress bar
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
