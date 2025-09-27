# Agent Validation: Gaussian Slicer (Metal/SwiftPM)

This workflow validates that the generated Swift Package Manager (SwiftPM) project for the Gaussian Slicer compiles successfully on macOS. This confirms that both the Swift code and the Metal shaders are valid and correctly configured for the target environment.

## Prerequisites

* **Environment:** macOS 13.0+ (as specified in `Package.swift`).
* **Tools:** Xcode Command Line Tools (includes the Swift toolchain and Metal compiler).

## Validation Steps

1.  **Execute Bootstrap Script**

    Ensure the bootstrap script runs successfully and creates the project.
    ```python
    agwa.shell.run("./bootstrap_gaussianslicer.sh")
    ```

2.  **Navigate to Project Directory**

    Change the current working directory to the root of the generated project.
    ```python
    agwa.shell.run("cd GaussianSlicer")
    ```

3.  **Clean Previous Builds**

    Ensure a fresh build environment.
    ```python
    agwa.shell.run("swift package clean")
    ```

4.  **Build the Project (Debug Configuration)**

    Execute the Swift build command. SwiftPM automatically invokes the Metal compiler (`metal`) to compile `Shaders.metal` into a `default.metallib` and links it with the Swift code. A failure in either compilation stage will cause this step to fail.
    ```python
    # Set the expectation for the command to succeed
    agwa.shell.run("swift build -c debug")
    ```

5.  **Build the Project (Release Configuration)**

    Verify that the project builds with release optimizations enabled. This is the configuration optimized for performance on the M1 Pro.
    ```python
    # Set the expectation for the command to succeed
    agwa.shell.run("swift build -c release")
    ```

6.  **Verify Executable Creation**

    Check that the final executable binary was created in the release build directory.
    ```python
    # The exact path depends on the architecture (e.g., arm64-apple-macosx).
    # We use a wildcard search to find the executable robustly.
    agwa.shell.run("find .build -type f -name 'GaussianSlicer' -path '*/release/*' | grep .")
    ```

7.  **Cleanup**

    Navigate back to the original directory.
    ```python
    agwa.shell.run("cd ..")
    ```

## Success Criteria

The validation is successful if all steps complete without error (exit code 0), confirming successful compilation of the Swift and Metal code.
