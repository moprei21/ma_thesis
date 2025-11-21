import os
from datetime import datetime

class LLMReporter:
    def __init__(self, base_dir="synthetic_data", use_timestamps=True):
        self.base_dir = base_dir
        print(f"Reports will be saved to: {self.base_dir}")
        self.use_timestamps = use_timestamps
        self.files = {}  # name -> file path

        os.makedirs(self.base_dir, exist_ok=True)

    def register_file(self, name, filename=None, ext="txt"):
        """Register a file with a logical name"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.use_timestamps else ""
            filename = f"{name}_{timestamp}.{ext}" if timestamp else f"{name}.{ext}"
        filepath = os.path.join(self.base_dir, filename)
        self.files[name] = filepath

    def write(self, name, content, newline=True):
        """Write content to a registered file"""
        if name not in self.files:
            raise ValueError(f"No file registered under name '{name}'")

        mode = "a"
        with open(self.files[name], mode, encoding="utf-8") as f:
            f.write(content)
            if newline:
                f.write("\n")

    def get_filepath(self, name):
        """Get the full path of a registered file"""
        return self.files.get(name, None)
    

