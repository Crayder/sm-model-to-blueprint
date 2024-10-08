import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox, scrolledtext
from tkinter import ttk
import os
import sys
import json
import numpy as np
import uuid
import shutil

# Import modules
import obj2vox
import constants
from help import MarkdownHelpViewer

# Function to get the blueprints directory
def get_blueprints_directory():
    blueprint_dir = os.path.join(os.getenv('APPDATA'), 'Axolot Games', 'Scrap Mechanic', 'User')
    if not os.path.exists(blueprint_dir):
        return None

    user_dirs = [d for d in os.listdir(blueprint_dir) if d.startswith('User_')]
    if not user_dirs:
        return None

    blueprint_dir = os.path.join(blueprint_dir, user_dirs[0], 'Blueprints')
    if os.path.exists(blueprint_dir):
        return blueprint_dir
    else:
        return None

# Function to get absolute path to resource, works for dev and for PyInstaller.
def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")  # Use the current directory for development
    return os.path.join(base_path, relative_path)

class Redirector:
    def __init__(self, text_widget, original_stream):
        self.text_widget = text_widget
        self.original_stream = original_stream

    def write(self, text):
        if self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()
        
        self.text_widget.after(0, self.append_text, text)

    def append_text(self, text):
        self.text_widget.configure(state='normal')
        self.text_widget.insert('end', text)
        self.text_widget.see('end')  # Auto-scroll to the end
        self.text_widget.configure(state='disabled')

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()

class VoxelConverterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OBJ to Vox Converter")
        self.create_widgets()

        # Save original stdout and stderr
        self.original_stdout = sys.stdout if sys.stdout else None
        self.original_stderr = sys.stderr if sys.stderr else None

    def create_widgets(self):
        # Determine the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Main frame to hold the input and help frames
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        main_frame.grid_columnconfigure(0, weight=1)

        # Input File Selection Frame
        input_frame = ttk.LabelFrame(main_frame, text="Input OBJ File")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Handle default input file path. If in dev mode, use constants.CONFIG_VALUES["input_file"], otherwise use an empty string.
        if getattr(sys, 'frozen', False):
            default_input = ""
        else:
            default_input = constants.CONFIG_VALUES["input_file"]
            if not os.path.isabs(default_input):
                default_input = resource_path(default_input)
        self.input_path = tk.StringVar(value=default_input)

        input_entry = ttk.Entry(input_frame, textvariable=self.input_path, width=50)
        input_entry.grid(row=0, column=0, padx=5, pady=5)

        input_button = ttk.Button(input_frame, text="Browse", command=self.browse_input)
        input_button.grid(row=0, column=1, padx=5, pady=5)

        # Help Button Frame (separate from input frame, aligned to the right)
        help_frame = ttk.Frame(main_frame)
        help_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")  # Adjust alignment with "ne" for top-right

        # Help button inside its own frame
        help_button = ttk.Button(help_frame, text="Help", command=self.open_help, width=10)
        help_button.pack(fill=tk.Y, expand=True)

        # Output File Selection
        output_frame = ttk.LabelFrame(self.root, text="Output Blueprint JSON File")
        output_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # Handle default output file path
        default_output = constants.CONFIG_VALUES["output_file"]
        if not os.path.isabs(default_output):
            default_output = os.path.abspath(os.path.join(script_dir, default_output))
        self.output_path = tk.StringVar(value=default_output)

        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=50)
        self.output_entry.grid(row=0, column=0, padx=5, pady=5)

        self.output_button = ttk.Button(output_frame, text="Browse", command=self.browse_output)
        self.output_button.grid(row=0, column=1, padx=5, pady=5)

        # New Blueprint Checkbox
        self.new_blueprint_var = tk.BooleanVar(value=False)
        self.new_blueprint_check = ttk.Checkbutton(
            output_frame,
            text="New Blueprint",
            variable=self.new_blueprint_var,
            command=self.toggle_new_blueprint
        )
        self.new_blueprint_check.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # Output Name Entry (hidden by default)
        self.output_name_label = ttk.Label(output_frame, text="Output Name:")
        self.output_name_entry = ttk.Entry(output_frame, width=30)

        # Transformation Parameters
        transform_frame = ttk.LabelFrame(self.root, text="Model Transformations")
        transform_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        # Scaling
        scaling_label = ttk.Label(transform_frame, text="Scaling:")
        scaling_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.scale_x = tk.DoubleVar(value=constants.CONFIG_VALUES["obj_scale"])
        self.scale_y = tk.DoubleVar(value=constants.CONFIG_VALUES["obj_scale"])
        self.scale_z = tk.DoubleVar(value=constants.CONFIG_VALUES["obj_scale"])

        ttk.Label(transform_frame, text="X:").grid(row=0, column=1, padx=5, pady=5, sticky="e")
        scale_x_entry = ttk.Entry(transform_frame, textvariable=self.scale_x, width=10)
        scale_x_entry.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(transform_frame, text="Y:").grid(row=0, column=3, padx=5, pady=5, sticky="e")
        scale_y_entry = ttk.Entry(transform_frame, textvariable=self.scale_y, width=10, state="disabled")
        scale_y_entry.grid(row=0, column=4, padx=5, pady=5)

        ttk.Label(transform_frame, text="Z:").grid(row=0, column=5, padx=5, pady=5, sticky="e")
        scale_z_entry = ttk.Entry(transform_frame, textvariable=self.scale_z, width=10, state="disabled")
        scale_z_entry.grid(row=0, column=6, padx=5, pady=5)

        # **Add Uniform Scaling Checkbox**
        self.uniform_scaling = tk.BooleanVar(value=True)
        uniform_check = ttk.Checkbutton(
            transform_frame,
            text="Uniform",
            variable=self.uniform_scaling,
            command=self.toggle_uniform_scaling
        )
        uniform_check.grid(row=0, column=7, padx=5, pady=5, sticky="w")

        # Bind Scale X changes for use when Uniform is Enabled
        self.scale_x.trace_add("write", self.on_scale_x_change)

        # Offset
        offset_label = ttk.Label(transform_frame, text="Offset:")
        offset_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.offset_x = tk.DoubleVar(value=constants.CONFIG_VALUES["obj_offset"][0])
        self.offset_y = tk.DoubleVar(value=constants.CONFIG_VALUES["obj_offset"][1])
        self.offset_z = tk.DoubleVar(value=constants.CONFIG_VALUES["obj_offset"][2])

        ttk.Label(transform_frame, text="X:").grid(row=1, column=1, padx=5, pady=5, sticky="e")
        offset_x_entry = ttk.Entry(transform_frame, textvariable=self.offset_x, width=10)
        offset_x_entry.grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(transform_frame, text="Y:").grid(row=1, column=3, padx=5, pady=5, sticky="e")
        offset_y_entry = ttk.Entry(transform_frame, textvariable=self.offset_y, width=10)
        offset_y_entry.grid(row=1, column=4, padx=5, pady=5)

        ttk.Label(transform_frame, text="Z:").grid(row=1, column=5, padx=5, pady=5, sticky="e")
        offset_z_entry = ttk.Entry(transform_frame, textvariable=self.offset_z, width=10)
        offset_z_entry.grid(row=1, column=6, padx=5, pady=5)

        # Rotation
        rotation_label = ttk.Label(transform_frame, text="Rotation:")
        rotation_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.rotate_axis = tk.StringVar(value=constants.CONFIG_VALUES["rotate_axis"].lower())
        self.rotate_angle = tk.DoubleVar(value=constants.CONFIG_VALUES["rotate_angle"])

        axis_options = ['x', 'y', 'z']
        axis_dropdown = ttk.Combobox(transform_frame, textvariable=self.rotate_axis, values=axis_options, state="readonly", width=5)
        axis_dropdown.grid(row=2, column=1, padx=5, pady=5)
        axis_dropdown.current(axis_options.index(constants.CONFIG_VALUES["rotate_axis"].lower()))

        angle_entry = ttk.Entry(transform_frame, textvariable=self.rotate_angle, width=10)
        angle_entry.grid(row=2, column=2, padx=5, pady=5)
        ttk.Label(transform_frame, text="degrees").grid(row=2, column=3, padx=5, pady=5, sticky="w")

        # Color Selection
        appearance_frame = ttk.LabelFrame(self.root, text="Block Appearance")
        appearance_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.use_set_color = tk.BooleanVar(value=constants.CONFIG_VALUES["use_set_color"])
        set_color_check = ttk.Checkbutton(appearance_frame, text="Use Custom Set Color", variable=self.use_set_color, command=self.toggle_color_picker)
        set_color_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.set_color = constants.CONFIG_VALUES["set_color"]
        self.color_display = tk.Canvas(appearance_frame, width=120, height=20, bg=self.rgb_to_hex(self.set_color))
        self.color_display.grid(row=0, column=1, padx=5, pady=5)

        color_button = ttk.Button(appearance_frame, text="Choose Color", command=self.choose_color)
        color_button.grid(row=0, column=2, padx=5, pady=5)

        # Material Selection
        self.use_set_block = tk.BooleanVar(value=constants.CONFIG_VALUES["use_set_block"])
        set_block_check = ttk.Checkbutton(appearance_frame, text="Use Custom Set Block", variable=self.use_set_block, command=self.toggle_material_dropdown)
        set_block_check.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.set_block = tk.StringVar(value=constants.CONFIG_VALUES["set_block"])
        self.block_dropdown = ttk.Combobox(appearance_frame, textvariable=self.set_block, values=list(constants.BLOCK_IDS.keys()), state="disabled", width=16)
        self.block_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.block_dropdown.set(constants.CONFIG_VALUES["set_block"])

        # Additional Options
        options_frame = ttk.LabelFrame(self.root, text="Additional Options")
        options_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        self.use_scrap_colors = tk.BooleanVar(value=constants.CONFIG_VALUES["use_scrap_colors"])
        scrap_colors_check = ttk.Checkbutton(options_frame, text="Use Scrap Colors", variable=self.use_scrap_colors)
        scrap_colors_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.vary_colors = tk.BooleanVar(value=constants.CONFIG_VALUES["vary_colors"])
        vary_colors_check = ttk.Checkbutton(options_frame, text="Vary Colors", variable=self.vary_colors)
        vary_colors_check.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.interior_fill = tk.BooleanVar(value=constants.CONFIG_VALUES["interior_fill"])
        interior_fill_check = ttk.Checkbutton(options_frame, text="Interior Fill", variable=self.interior_fill)
        interior_fill_check.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Voxel Scale
        voxel_scale_frame = ttk.LabelFrame(self.root, text="Voxel Scale")
        voxel_scale_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        self.voxel_scale = tk.DoubleVar(value=constants.CONFIG_VALUES["voxel_scale"])
        voxel_scale_entry = ttk.Entry(voxel_scale_frame, textvariable=self.voxel_scale, width=10)
        voxel_scale_entry.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(voxel_scale_frame, text="units").grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Run Button
        self.run_button = ttk.Button(self.root, text="Run Conversion", command=self.run_conversion)
        self.run_button.grid(row=7, column=0, padx=10, pady=20)

        # Console Output
        console_frame = ttk.LabelFrame(self.root, text="Console Output")
        console_frame.grid(row=8, column=0, padx=10, pady=10, sticky="nsew")

        self.console_text = tk.Text(console_frame, height=10, wrap='word', state='disabled')
        self.console_text.pack(side='left', fill='both', expand=True)

        console_scrollbar = ttk.Scrollbar(console_frame, command=self.console_text.yview)
        console_scrollbar.pack(side='right', fill='y')

        self.console_text['yscrollcommand'] = console_scrollbar.set

        # Make sure the grid expands properly
        for i in range(9):
            self.root.grid_rowconfigure(i, weight=0)
        self.root.grid_rowconfigure(8, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def open_help(self):
        if not hasattr(self, 'help_window') or not self.help_window.winfo_exists():
            self.help_window = MarkdownHelpViewer(self.root, help_file=resource_path("doc/help.md"))  # Adjust the help file path if needed

    def browse_input(self):
        default_input = constants.CONFIG_VALUES["input_file"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(default_input):
            default_input = os.path.abspath(os.path.join(script_dir, default_input))
        initial_dir = os.path.dirname(default_input) if default_input else os.getcwd()
        initial_file = os.path.basename(default_input) if default_input else ""
        
        file_path = filedialog.askopenfilename(
            title="Select OBJ File",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("OBJ Files", "*.obj"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_path.set(file_path)

    def browse_output(self):
        default_output = constants.CONFIG_VALUES["output_file"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(default_output):
            default_output = os.path.abspath(os.path.join(script_dir, default_output))
        initial_dir = os.path.dirname(default_output) if default_output else os.getcwd()
        initial_file = os.path.basename(default_output) if default_output else "blueprint.json"
        
        file_path = filedialog.asksaveasfilename(
            title="Select Output JSON File",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.output_path.set(file_path)

    def toggle_new_blueprint(self):
        if self.new_blueprint_var.get():
            # Disable output file selector
            self.output_entry.configure(state='disabled')
            self.output_button.configure(state='disabled')
            
            # Show Output Name entry
            self.output_name_label.grid(row=1, column=1, padx=5, pady=5, sticky="e")
            self.output_name_entry.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        else:
            # Enable output file selector
            self.output_entry.configure(state='normal')
            self.output_button.configure(state='normal')
            
            # Hide Output Name entry
            self.output_name_label.grid_remove()
            self.output_name_entry.grid_remove()

    def toggle_uniform_scaling(self):
        if self.uniform_scaling.get():
            # Set Y and Z to X's value
            self.scale_y.set(self.scale_x.get())
            self.scale_z.set(self.scale_x.get())
            # Disable Y and Z entries
            self.disable_scale_entries()
        else:
            # Enable Y and Z entries
            self.enable_scale_entries()

    def disable_scale_entries(self):
        # Disable Y and Z scale entry widgets
        # Locate the transform_frame
        for child in self.root.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Model Transformations":
                transform_frame = child
                break
        else:
            return  # transform_frame not found

        # Disable Y and Z entry widgets
        for widget in transform_frame.winfo_children():
            if isinstance(widget, ttk.Entry):
                var = widget.cget("textvariable")
                if var == str(self.scale_y) or var == str(self.scale_z):
                    widget.configure(state='disabled')

    def enable_scale_entries(self):
        # Enable Y and Z scale entry widgets
        # Locate the transform_frame
        for child in self.root.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Model Transformations":
                transform_frame = child
                break
        else:
            return  # transform_frame not found

        # Enable Y and Z entry widgets
        for widget in transform_frame.winfo_children():
            if isinstance(widget, ttk.Entry):
                var = widget.cget("textvariable")
                if var == str(self.scale_y) or var == str(self.scale_z):
                    widget.configure(state='normal')

    def on_scale_x_change(self, *args):
        if self.uniform_scaling.get():
            # Update Y and Z to match X
            new_x = self.scale_x.get()
            self.scale_y.set(new_x)
            self.scale_z.set(new_x)

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Choose Set Color", initialcolor=self.rgb_to_hex(self.set_color))
        if color_code[1]:
            self.color_display.configure(bg=color_code[1])
            self.set_color = self.hex_to_rgb(color_code[1])

    def toggle_color_picker(self):
        if self.use_set_color.get():
            self.color_display.configure(state="normal")
        else:
            self.color_display.configure(state="disabled")

    def toggle_material_dropdown(self):
        if self.use_set_block.get():
            self.block_dropdown.configure(state="readonly")
        else:
            self.block_dropdown.configure(state="disabled")

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]

    def run_conversion(self):
        # Validate inputs
        if not self.input_path.get():
            messagebox.showerror("Input Error", "Please select an input OBJ file.")
            return
        if not self.new_blueprint_var.get() and not self.output_path.get():
            messagebox.showerror("Input Error", "Please select an output JSON file.")
            return
        if self.new_blueprint_var.get() and not self.output_name_entry.get().strip():
            messagebox.showerror("Input Error", "Please enter an output name for the new blueprint.")
            return

        input_file = self.input_path.get()

        if self.new_blueprint_var.get():
            # New Blueprint mode
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(script_dir, "blueprint.json")
            output_name = self.output_name_entry.get().strip()
        else:
            # Regular mode
            output_file = self.output_path.get()
            output_name = None

        # Further validation of file paths
        if not os.path.isfile(input_file) or not input_file.endswith(".obj"):
            messagebox.showerror("Input Error", "Invalid input file selected.")
            return

        # Make them absolute paths if not already
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(input_file):
            input_file = os.path.abspath(os.path.join(script_dir, input_file))
        if not self.new_blueprint_var.get() and not os.path.isabs(output_file):
            output_file = os.path.abspath(os.path.join(script_dir, output_file))

        voxel_scale = self.voxel_scale.get()
        obj_scale = np.array([self.scale_x.get(), self.scale_y.get(), self.scale_z.get()])
        obj_offset = np.array([self.offset_x.get(), self.offset_y.get(), self.offset_z.get()])
        rotate_axis = self.rotate_axis.get()
        rotate_angle = self.rotate_angle.get()

        use_set_color = self.use_set_color.get()
        set_color = self.set_color if use_set_color else constants.FALLBACK_MATERIAL["color"]

        use_set_block = self.use_set_block.get()
        set_block = self.set_block.get() if use_set_block else "plastic"

        use_scrap_colors = self.use_scrap_colors.get()
        vary_colors = self.vary_colors.get()
        interior_fill = self.interior_fill.get()

        # Prepare parameters
        params = {
            "input_file": input_file,
            "output_file": output_file,
            "voxel_scale": voxel_scale,
            "obj_scale": obj_scale,
            "obj_offset": obj_offset,
            "rotate_axis": rotate_axis,
            "rotate_angle": rotate_angle,
            "use_set_color": use_set_color,
            "set_color": set_color,
            "use_set_block": use_set_block,
            "set_block": set_block,
            "use_scrap_colors": use_scrap_colors,
            "vary_colors": vary_colors,
            "interior_fill": interior_fill,
            "new_blueprint": self.new_blueprint_var.get(),
            "output_name": output_name
        }

        # Disable the Run button to prevent multiple clicks
        self.run_button.configure(state='disabled')

        # Run the conversion in a separate thread to keep the UI responsive
        import threading
        thread = threading.Thread(target=self.execute_conversion, args=(params,))
        thread.start()

    def execute_conversion(self, params):
        # Redirect stdout and stderr to the console_text widget
        sys.stdout = Redirector(self.console_text, self.original_stdout)
        sys.stderr = Redirector(self.console_text, self.original_stderr)

        try:
            total_execution_time = obj2vox.main(
                input_file=params["input_file"],
                output_file=params["output_file"],
                voxel_scale=params["voxel_scale"],
                obj_scale=params["obj_scale"],
                obj_offset=params["obj_offset"],
                rotate_axis=params["rotate_axis"],
                rotate_angle=params["rotate_angle"],
                use_set_color=params["use_set_color"],
                set_color=params["set_color"],
                use_set_block=params["use_set_block"],
                set_block=params["set_block"],
                use_scrap_colors=params["use_scrap_colors"],
                vary_colors=params["vary_colors"],
                interior_fill=params["interior_fill"]
            )
            if params["new_blueprint"]:
                blueprint_path = params["output_file"]
                if os.path.exists(blueprint_path):
                    self.createBlueprint(blueprint_path, params["output_name"])
            self.console_text.after(0, lambda: messagebox.showinfo(
                "Success", f"Conversion completed successfully!\n\nOutput saved to:\n{params['output_file']}\n\nTotal execution time: {total_execution_time:.2f} seconds"))
        except Exception as e:
            self.console_text.after(0, lambda e=e: messagebox.showerror(
                "Error", f"An error occurred during conversion:\n{str(e)}"))
        finally:
            # Restore original stdout and stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            # Re-enable the Run button
            self.run_button.configure(state='normal')

    def createBlueprint(self, json_path, output_name):
        blueprint_id = str(uuid.uuid4())
        blueprint_dir = get_blueprints_directory()
        if not blueprint_dir:
            messagebox.showerror("Blueprint Error", "Could not find the Scrap Mechanic blueprints directory.\n\nLeaving blueprint.json in the app folder.")
            return

        # Create the blueprint directory in the user's blueprints folder
        blueprint_path = os.path.join(blueprint_dir, blueprint_id)
        os.makedirs(blueprint_path, exist_ok=True)

        # Copy the JSON file to the blueprint directory, as blueprint.json
        blueprint_json = os.path.join(blueprint_path, "blueprint.json")
        shutil.copy(json_path, blueprint_json)

        # Copy blueprint_schematic/icon.png to the blueprint directory
        icon_path = resource_path("blueprint_schematic/icon.png")
        if os.path.exists(icon_path):
            icon_dest = os.path.join(blueprint_path, "icon.png")
            shutil.copy(icon_path, icon_dest)
        
        # Load blueprint_schematic/description.json
        desc_path = resource_path("blueprint_schematic/description.json")
        with open(desc_path, 'r') as f:
            desc_data = json.load(f)

        # Update the description.json with the output name
        desc_data["name"] = output_name
        desc_data["localId"] = blueprint_id

        # Save the updated description.json to the blueprint directory
        desc_dest = os.path.join(blueprint_path, "description.json")
        with open(desc_dest, 'w') as f:
            json.dump(desc_data, f, indent=4)

        # Delete the original JSON file
        os.remove(json_path)

        print(f"Blueprint created successfully: {output_name} ({blueprint_id})")

def main():
    root = tk.Tk()
    app = VoxelConverterUI(root)
    dirname = os.path.dirname(__file__)
    icon_path = os.path.join(dirname, "blueprint_schematic", "icon.png")
    if os.path.exists(icon_path):
        root.iconphoto(True, tk.PhotoImage(file=icon_path))
    root.title("Model Scrapper")
    root.mainloop()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
