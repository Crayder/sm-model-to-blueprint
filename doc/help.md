# Model Scrapper Help

## Introduction

**Model Scrapper** is a tool that allows you to transform 3D models in **OBJ** format into blueprints compatible with the game **Scrap Mechanic**. This enables you to import custom models into the game and build them using in-game materials.

- **Note:** You can choose the blocks and colors to use in Blender using materials. See more details in the "Using Materials in Blender" section at the bottom.

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
- **Note**: By default this rotates 90 degrees around the X-axis. This is the correct rotation to result in an upright model in Scrap Mechanic when using an OBJ file. Later support for other formats may require different rotations.

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


---

## Using Materials in Blender

By default, if you just have no material or downloaded a model from the internet, the converter will use the default block and color for the entire model.

To define which blocks and colors the converter will use, you need to create materials in Blender for the objects in your scene. Here's how to do it:

1. **Create Materials**: In Blender, assign materials to the different parts of your object.
2. **Material Names**: The material names will determine which block to use. You can choose a block name from the list below.
3. **Base Colors**: The base color of each material will determine the color applied to the blocks in the final output.
4. **Assign to Faces**: Assign these materials to the faces in your Blender model. The converter will use the specified block and color for those faces during the conversion.

### Block Names

- scrapwood
- wood1, wood2, wood3
- scrapmetal,
- metal1, metal2, metal3
- scrapstone
- concrete1, concrete2, concrete3
- cardboard,
- sand,
- plastic,
- glass,
- glasstile,
- armoredglass,
- bubblewrap,
- restroom,
- tiles,
- bricks,
- lights,
- caution,
- crackedconcrete,
- concretetiles,
- metalbricks,
- beam,
- insulation,
- drywall,
- carpet,
- plasticwall,
- metalnet,
- crossnet,
- tryponet,
- stripednet,
- squarenet,
- spaceshipmetal,
- spaceshipfloor,
- treadplate,
- warehousefloor,
- wornmetal,
- framework,
- challenge01, challenge02
- challengeglass
