1. After generating the voxels, before decomposition and before filling. Detect whether the center of the voxel grid isn't too high for the lift to place the object. If it is, then ask the user if they wish to proceed with the conversion. If they do, then proceed with the conversion. If they don't, then stop the conversion and return to the main screen. (This is to prevent the user from converting a model that is too high for the lift to place in the game. But, if the user has Better Lifts mod installed, then they may be able to place it anyways.) Perhaps also allow the user to request that we add a plane across the bottom of the model so the lift would be at the lowest point of the model. Options Proceed, Cancel, and Add Plane.

2. The classify_voxels_raycast centers/direction process can take a long time for large models, need to either:
        1. Add a progress bar for the voxel centers and ray directions calculations.
        2. Incorporate the voxel centers and ray directions calculations into the ray casting process.

3. Add saving/loading of the user's input to a config.json file from the executable's directory. This way the user's preferences are saved between sessions. Also, add a reset button to reset the settings to default values.

4. Need to consider batch methods for filling. Like:
        - Slice the model into smaller parts and convert each part to separate blueprints. Weld those together in-game manually. Like layers of the model, so we could still do XY plane raycasting and determine whether any point is inside the model.
        - Perform the normal non-fill method first. Store that in a temporary file. Then for filling process one section or layer of that file at a time, after processing that section/layer output data to a separate temporary file. After processing the entire model, output the data to a blueprint file. This would allow for the processing of large models without running out of memory.
        - Perhaps offer both of those ideas as different options that can be enabled separately or together.
