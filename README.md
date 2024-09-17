# Model Scrapper

## Introduction

The **Model Scrapper** is a tool designed to transform 3D models in OBJ format into blueprints compatible with the game **Scrap Mechanic**. It provides a user-friendly interface to customize various aspects of the conversion process, allowing for detailed and accurate imports of custom models into the game.

---

## Features

- **OBJ to Blueprint Conversion**: Converts OBJ files into Scrap Mechanic blueprints.
- **Model Transformation Options**: Scale, offset, and rotate models before conversion.
- **Custom Block Materials and Colors**: Choose specific blocks and colors for your model.
- **Scrap Mechanic Color Palette Support**: Use colors from the game's palette.
- **Interior Filling**: Option to fill the interior of models to make them solid.
- **User-Friendly GUI**: Easy-to-use interface built with Tkinter.

---

## Usage

**Run the Application**:

Each of the modules can be run individually, but the main application is `launcher.py`. Run the following command to start the application:

   ```bash
   python launcher.py
   ```

---

## Building an Executable

1. **Install PyInstaller**:

   ```bash
   pip install pyinstaller
   ```

2. **Install Project Dependencies**:

   Install the required packages using pip:

   ```bash
   pip install numpy trimesh scipy tqdm
   ```

3. **Build the Executable**:
   
   Navigate to the project directory and run the following command:

   ```bash
   pyinstaller --onefile --windowed --icon=icon.ico --add-data "blueprint_schematic;blueprint_schematic" --add-data "doc;doc" launcher.py
   ```

This will create a single executable file in the `dist` directory.
