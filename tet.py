import tkinter as tk
from tkinter import ttk
import markdown2

class MarkdownHelpViewer(tk.Toplevel):
    def __init__(self, master=None, help_file="help.md"):
        super().__init__(master)
        self.title("Help Documentation")
        self.geometry("800x600")
        self.help_file = help_file

        # Create a Frame for Text and Scrollbars
        text_frame = ttk.Frame(self)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create Vertical Scrollbar
        self.v_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Create Text Widget
        self.text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=self.v_scroll.set,
            state=tk.NORMAL
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        # Configure Scrollbars
        self.v_scroll.config(command=self.text.yview)

        # Load and Render Markdown Content
        self.load_markdown()

    def load_markdown(self):
        try:
            with open(self.help_file, "r", encoding="utf-8") as file:
                markdown_content = file.read()

            # Convert Markdown to HTML
            html_content = markdown2.markdown(markdown_content)

            # Use simple tags to convert HTML to text widget format (could be expanded)
            self.text.insert(tk.END, html_content)

            # Disable editing
            self.text.config(state=tk.DISABLED)

        except FileNotFoundError:
            self.text.insert(tk.END, "Error: Help file not found.")

def main():
    root = tk.Tk()
    root.title("My Tkinter App")
    root.geometry("300x200")

    def open_help():
        help_window = MarkdownHelpViewer(root)

    # Add Help Button
    help_button = ttk.Button(root, text="Help", command=open_help)
    help_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
