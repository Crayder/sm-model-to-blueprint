# OBJ to Scrap Mechanic Blueprint Converter - Project README

## Introduction

The **OBJ to Scrap Mechanic Blueprint Converter** is a tool designed to transform 3D models in OBJ format into blueprints compatible with the game **Scrap Mechanic**. It provides a user-friendly interface to customize various aspects of the conversion process, allowing for detailed and accurate imports of custom models into the game.

---

## Features

- **OBJ to Blueprint Conversion**: Converts OBJ files into Scrap Mechanic blueprints.
- **Model Transformation Options**: Scale, offset, and rotate models during conversion.
- **Custom Block Materials and Colors**: Choose specific blocks and colors for your model.
- **Scrap Mechanic Color Palette Support**: Use colors from the game's palette.
- **Interior Filling**: Option to fill the interior of models to make them solid.
- **User-Friendly GUI**: Easy-to-use interface built with Tkinter.

---

## Requirements

- **Python 3.x**
- **Required Python Packages**:
  - `numpy`
  - `trimesh`
  - `scipy`
  - `tqdm`

### Installing Dependencies

Install the required packages using pip:

```bash
pip install numpy trimesh scipy tqdm
```

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd yourrepository
   ```

3. **Install Dependencies**:

   ```bash
   pip install numpy trimesh scipy tqdm
   ```

---

## Usage

1. **Run the Application**:

   ```bash
   python launcher.py
   ```

2. **Follow the User Guide**:

   Configure your settings and run the conversion as detailed in the [User Guide](#using-the-app).

---

## Project Structure

### Files Overview

- **launcher.py**: The main application file that provides the GUI and handles user interactions.
- **obj2vox.py**: Contains the core conversion logic from OBJ to voxel representation and blueprint generation.
- **constants.py**: Defines constants, block IDs, color palettes, and default configuration values.

### File Descriptions

#### launcher.py

- **Purpose**: Provides the graphical user interface for the converter.
- **Key Functions**:
  - **create_widgets**: Sets up the GUI components.
  - **run_conversion**: Gathers parameters and initiates the conversion process.
  - **execute_conversion**: Runs the conversion in a separate thread and handles output redirection.

#### obj2vox.py

- **Purpose**: Handles the conversion logic from OBJ files to Scrap Mechanic blueprints.
- **Key Functions**:
  - **parse_obj_file**: Reads the OBJ file and extracts vertices and faces.
  - **create_voxel_grid**: Generates a voxel grid based on the model.
  - **mark_voxels**: Determines which voxels are occupied by the model.
  - **minimal_prism_decomposition**: Optimizes the voxel grid into prisms for blueprint generation.
  - **save_to_json_file**: Outputs the final blueprint JSON file.

#### constants.py

- **Purpose**: Stores constants and default values used across the application.
- **Contents**:
  - **BLOCK_IDS**: Mapping of block names to their UUIDs in Scrap Mechanic.
  - **SM_COLORS**: Scrap Mechanic's color palette.
  - **CONFIG_VALUES**: Default configuration settings for the application.
  - **FALLBACK_MATERIAL**: Default material settings if none are specified.

---

## Building an Executable

To distribute the application without requiring users to install Python or dependencies, you can build an executable using **PyInstaller**.

### Steps to Build

1. **Install PyInstaller**:

   ```bash
   pip install pyinstaller
   ```

2. **Build the Executable**:

   ```bash
   pyinstaller --onefile --windowed launcher.py
   ```

   - The `--onefile` flag creates a single executable file.
   - The `--windowed` flag prevents a console window from appearing.

3. **Include Resource Files**:

   Ensure that any required resource files (e.g., images, default configurations) are included in the build. You may need to adjust the `--add-data` option in PyInstaller.

   Example:

   ```bash
   pyinstaller --onefile --windowed --add-data "blueprint_schematic;blueprint_schematic" launcher.py
   ```

4. **Distribute**:

   The executable will be located in the `dist` directory. You can distribute this file to users.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1. **Fork the Repository**
2. **Create a Feature Branch**:

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**
4. **Push to Your Fork**
5. **Submit a Pull Request**

---

## License

[Specify your project's license here, e.g., MIT License]

---

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)

---

## Acknowledgments

- Thanks to the contributors of open-source libraries used in this project.
- Inspired by the Scrap Mechanic community's creativity and innovation.
