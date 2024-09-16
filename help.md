# OBJ to Scrap Mechanic Blueprint Converter - User Guide

## Introduction

The **OBJ to Scrap Mechanic Blueprint Converter** is a tool that allows you to transform 3D models in OBJ format into blueprints compatible with the game **Scrap Mechanic**. This enables you to import custom models into the game and build them using in-game materials.

---

## Using the App

### 1. Input OBJ File

- **Purpose**: Select the OBJ file you wish to convert.
- **How to Use**:
  - Click the **Browse** button in the **Input OBJ File** section.
  - Navigate to the location of your OBJ file and select it.

### 2. Output Blueprint JSON File

- **Purpose**: Specify where the output blueprint JSON file will be saved.
- **How to Use**:
  - Click the **Browse** button in the **Output Blueprint JSON File** section.
  - Choose the desired save location and filename.

### 3. New Blueprint Option

- **Purpose**: Create a new blueprint directly in your Scrap Mechanic blueprints folder.
- **How to Use**:
  - Check the **New Blueprint** checkbox.
  - An **Output Name** field will appear. Enter the name for your new blueprint.
  - The output file selector will be disabled since the blueprint will be saved automatically.

### 4. Model Transformations

#### a. Scaling

- **Purpose**: Adjust the size of your model along the X, Y, and Z axes.
- **How to Use**:
  - Enter scaling values in the **X**, **Y**, and **Z** fields under **Scaling**.
  - **Uniform Scaling**:
    - Check the **Uniform** checkbox to apply the same scaling factor to all axes.
    - When enabled, changing the **X** value will automatically update **Y** and **Z**.

#### b. Offset

- **Purpose**: Shift the position of your model in 3D space.
- **How to Use**:
  - Enter offset values in the **X**, **Y**, and **Z** fields under **Offset**.

#### c. Rotation

- **Purpose**: Rotate your model around a specific axis.
- **How to Use**:
  - Select the rotation axis (**X**, **Y**, or **Z**) from the dropdown menu under **Rotation**.
  - Enter the rotation angle in degrees.

### 5. Block Appearance

#### a. Use Custom Set Color

- **Purpose**: Apply a specific color to all blocks in your model.
- **How to Use**:
  - Check the **Use Custom Set Color** checkbox.
  - Click **Choose Color** to open the color picker and select your desired color.
  - The selected color will be displayed next to the button.

#### b. Use Custom Set Block

- **Purpose**: Choose a specific block material for your model.
- **How to Use**:
  - Check the **Use Custom Set Block** checkbox.
  - Select the block type from the dropdown menu.

### 6. Additional Options

#### a. Use Scrap Colors

- **Purpose**: Replace your model's colors with the closest matching colors from Scrap Mechanic's palette.
- **How to Use**:
  - Check the **Use Scrap Colors** checkbox.

#### b. Vary Colors

- **Purpose**: Slightly vary the colors of the blocks for a more natural appearance.
- **How to Use**:
  - Check the **Vary Colors** checkbox.

#### c. Interior Fill

- **Purpose**: Fill the interior of your model, making it solid instead of hollow.
- **How to Use**:
  - Check the **Interior Fill** checkbox.

### 7. Voxel Scale

- **Purpose**: Set the resolution of the voxel grid used during conversion.
- **How to Use**:
  - Enter a value in the **Voxel Scale** field.
  - **Note**: Smaller values result in higher detail but increase processing time.

### 8. Running the Conversion

- **How to Use**:
  - After configuring all settings, click the **Run Conversion** button.
  - The conversion process will begin, and progress will be displayed in the **Console Output** section (as well as errors if any occur).

---

## Tips and Best Practices

- **Optimize Your Model**: Simplify your OBJ model by reducing unnecessary polygons to improve conversion speed.
- **Scaling Appropriately**: Adjust scaling to match Scrap Mechanic's unit system for accurate sizing. 1x object scale in Scrap Mechanic is 1 block per unit.
- **Interior Fill Usage**: Use interior fill for models that need to be solid. This will result in less prisms for the game to render in most cases.
