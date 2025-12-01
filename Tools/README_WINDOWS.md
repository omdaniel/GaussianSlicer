# Compiling vdb_writer on Windows

The `vdb_writer` tool is required for exporting OpenVDB files from the Gaussian Slicer application. Since the provided binary is for macOS, you must compile it from source for Windows.

## Prerequisites

1.  **Visual Studio 2019 or 2022**: Ensure "Desktop development with C++" workload is installed.
2.  **CMake**: Install from [cmake.org](https://cmake.org/download/).
3.  **vcpkg**: A C++ package manager.

## Step 1: Install vcpkg and OpenVDB

1.  Clone vcpkg (if you haven't already):
    ```powershell
    # Example: Installing to C:\vcpkg
    git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
    cd C:\vcpkg
    .\bootstrap-vcpkg.bat
    ```
    *Note: If you install vcpkg to a different location, adjust the paths in the following steps accordingly.*

2.  Install OpenVDB (x64-windows):
    Since you are using the Visual Studio bundled vcpkg, we will use "Manifest Mode".
    I have created a `vcpkg.json` file in this directory. The dependencies will be automatically installed when you run CMake in the next step.
    You do **not** need to run `vcpkg install` manually.

## Step 2: Compile vdb_writer

1.  Open a PowerShell terminal in this `Tools` directory.

2.  Create a build directory:
    ```powershell
    mkdir build
    cd build
    ```

3.  Configure with CMake (replace `C:/vcpkg/scripts/buildsystems/vcpkg.cmake` with your actual vcpkg path):
    **Important:** If you ran CMake before, delete the `build` directory contents first to ensure a clean configuration.
    ```powershell
    cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
    ```

4.  Build the project:
    ```powershell
    cmake --build . --config Release
    ```

5.  Locate the executable:
    The `vdb_writer.exe` will be in `Release\vdb_writer.exe`.

6.  Copy `vdb_writer.exe` to the `Tools` folder (where this README is) so the application can find it.
    ```powershell
    copy Release\vdb_writer.exe ..\vdb_writer.exe
    ```

## Troubleshooting

*   **Missing DLLs**: If running `vdb_writer.exe` fails due to missing DLLs (e.g., `openvdb.dll`, `tbb.dll`), you need to copy them from `C:\vcpkg\installed\x64-windows\bin` to the same folder as `vdb_writer.exe`.
