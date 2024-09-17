import tkinter as tk
from tkinter import ttk
import markdown2
from tkhtmlview import HTMLScrolledText  # Import the HTML widget

class MarkdownHelpViewer(tk.Toplevel):
    def __init__(self, master=None, help_file="help.md"):
        super().__init__(master)
        self.title("Help Documentation")
        self.geometry("800x600")
        self.help_file = help_file

        # Create a Frame for HTML content
        html_frame = ttk.Frame(self)
        html_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create HTMLScrolledText widget which includes scrollbars
        self.html_widget = HTMLScrolledText(
            html_frame,
            html="",  # Initial empty content
            background="white",
            foreground="black",
            font=("Arial", 12),
            wrap="word"
        )
        self.html_widget.pack(fill=tk.BOTH, expand=True)

        # Load and Render Markdown Content
        self.load_markdown()

    def load_markdown(self):
        try:
            with open(self.help_file, "r", encoding="utf-8") as file:
                markdown_content = file.read()

            # Convert Markdown to HTML
            html_content = markdown2.markdown(markdown_content)

            # Set HTML content to the HTML widget
            self.html_widget.set_html(html_content)

        except FileNotFoundError:
            error_html = "<h1>Error</h1><p>Help file not found.</p>"
            self.html_widget.set_html(error_html)

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
