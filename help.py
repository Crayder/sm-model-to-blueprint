import tkinter as tk
from tkinter import ttk
from html.parser import HTMLParser
import markdown2

class HTMLToTkinterParser(HTMLParser):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        # Map the HTML tag to Tkinter tags
        if tag in ["h1", "h2", "h3", "h4"]:
            self.tag_stack.append(tag)
        elif tag == "b" or tag == "strong":
            self.tag_stack.append("bold")
        elif tag == "i" or tag == "em":
            self.tag_stack.append("italic")
        elif tag == "ul":
            self.tag_stack.append("list")
        elif tag == "ol":
            self.tag_stack.append("olist")
            self.list_counter = 1
        elif tag == "li":
            if self.tag_stack[-1] == "olist":
                self.text_widget.insert(tk.END, f"{self.list_counter}. ", "list")
                self.list_counter += 1
            else:
                self.text_widget.insert(tk.END, "• ", "list")
        elif tag == "br":
            self.text_widget.insert(tk.END, "\n")
        elif tag == "hr":
            self.text_widget.insert(tk.END, "\n" + "⎯" * 40 + "\n")

    def handle_endtag(self, tag):
        # Remove the tag from the stack
        if tag in ["h1", "h2", "h3", "h4", "b", "i", "strong", "em", "ul", "ol"]:
            self.tag_stack.pop()

    def handle_data(self, data):
        # Insert the text into the widget with the current tags
        if self.tag_stack:
            current_tag = self.tag_stack[-1]
            self.text_widget.insert(tk.END, data, current_tag)
        else:
            self.text_widget.insert(tk.END, data)

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
        self.text.config(font=("Helvetica", 12), spacing1=5, spacing3=5)

        # Define tags for different HTML elements (headers, bold, italic, etc.)
        self.text.tag_configure("h1", font=("Helvetica", 26, "bold"), spacing1=10, spacing3=10)
        self.text.tag_configure("h2", font=("Helvetica", 22, "bold"), spacing1=8, spacing3=8)
        self.text.tag_configure("h3", font=("Helvetica", 18, "bold"), spacing1=6, spacing3=6)
        self.text.tag_configure("h4", font=("Helvetica", 14, "bold"), spacing1=6, spacing3=6)
        self.text.tag_configure("bold", font=("Helvetica", 12, "bold"))
        self.text.tag_configure("italic", font=("Helvetica", 12, "italic"))
        self.text.tag_configure("list", lmargin1=25, lmargin2=50)

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
