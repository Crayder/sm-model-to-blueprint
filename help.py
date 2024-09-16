import tkinter as tk
from tkinter import ttk
from html.parser import HTMLParser
import markdown2

class HTMLToTkinterParser(HTMLParser):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.tag_stack = []
        self.list_counters = []
        self.previous_tag = None  # To keep track of the previous tag

    def handle_starttag(self, tag, attrs):
        if tag in ["h1", "h2", "h3", "h4", "p"]:
            self.tag_stack.append(tag)
        elif tag in ["b", "strong"]:
            self.tag_stack.append("bold")
        elif tag in ["i", "em"]:
            self.tag_stack.append("italic")
        elif tag in ["ul", "ol"]:
            self.tag_stack.append(tag)
            self.list_counters.append(0)
        elif tag == "li":
            list_type = self.tag_stack[-1]
            indent_level = len(self.list_counters)  # Each list increases the indent level
            indent = "\t" * indent_level  # Adjust this value to change indentation width
            if list_type == "ol":
                self.list_counters[-1] += 1
                number = self.list_counters[-1]
                self.text_widget.insert(tk.END, f"{indent}{number}. ", "list")
            else:
                self.text_widget.insert(tk.END, f"{indent}• ", "list")
        elif tag == "br":
            self.text_widget.insert(tk.END, "\n")
        elif tag == "hr":
            self.text_widget.insert(tk.END, "⎯" * 40 + "\n", "hr")
        # Add other tags as necessary

    def handle_endtag(self, tag):
        if tag in ["ul", "ol"]:
            self.list_counters.pop()
        if tag in ["h1", "h2", "h3", "h4", "p", "b", "i", "strong", "em", "ul", "ol"]:
            self.tag_stack.pop()
        # Add double newline after headers, paragraphs, etc.
        if tag in ["h1", "h2", "h3", "h4", "p"]:
            self.text_widget.insert(tk.END, "\n\n")
        # Add single newline after horizontal rules
        if tag == "hr":
            self.text_widget.insert(tk.END, "\n")
        # Add single newline after closing top-level lists
        if tag in ["ul", "ol"] and len(self.list_counters) == 0:
            self.text_widget.insert(tk.END, "\n")
        self.previous_tag = tag  # Update the previous tag

    def handle_data(self, data):
        # Strip leading and trailing newlines to manage newlines better
        data = data.strip("\n")
        if data:
            if self.tag_stack:
                current_tag = self.tag_stack[-1]
                self.text_widget.insert(tk.END, data, current_tag)
            else:
                self.text_widget.insert(tk.END, data)
            # Insert a newline after data if the current tag is a list item
            if self.tag_stack and self.tag_stack[-1] in ["ul", "ol"]:
                self.text_widget.insert(tk.END, "\n", "list")

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

        # Configure Styling
        self.configure_text()

        # Load and Render Markdown Content
        self.load_markdown()

    def configure_text(self):
        self.text.config(font=("Helvetica", 12), tabs=('10', '30'), spacing1=2, spacing3=2)  # Adjust spacing here

        # Define tags for different HTML elements (headers, bold, italic, etc.)
        self.text.tag_configure("h1", font=("Helvetica", 26, "bold", "underline"), spacing1=4, spacing3=4)
        self.text.tag_configure("h2", font=("Helvetica", 22, "bold"), spacing1=2, spacing3=2)
        self.text.tag_configure("h3", font=("Helvetica", 18, "bold"), spacing1=2, spacing3=2)
        self.text.tag_configure("h4", font=("Helvetica", 14, "bold"), spacing1=2, spacing3=2)
        self.text.tag_configure("bold", font=("Helvetica", 12, "bold"), spacing1=2, spacing3=2)
        self.text.tag_configure("italic", font=("Helvetica", 12, "italic"), spacing1=2, spacing3=2)
        self.text.tag_configure("list", lmargin1=10, lmargin2=20, font=("Helvetica", 12))
        self.text.tag_configure("hr", font=("Helvetica", 12))

    def load_markdown(self):
        try:
            with open(self.help_file, "r", encoding="utf-8") as file:
                markdown_content = file.read()

            # Convert Markdown to HTML
            html_content = markdown2.markdown(markdown_content)

            # Parse and display the HTML content
            parser = HTMLToTkinterParser(self.text)
            parser.feed(html_content)

            # Disable editing after rendering
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
