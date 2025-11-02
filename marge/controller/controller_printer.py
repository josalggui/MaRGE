from datetime import datetime
from pathlib import Path

import recon.data_processing as dp

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from PIL import Image as PILImage


class Printer:
    def __init__(self, main=None):
        self.main = main

    def add_text_to_story(self, text):
        if "ERROR" in text:
            text = f'<font color="red"><b>ERROR</b></font>{text[5:]}'
        elif "WARNING" in text:
            text = f'<font color="orange"><b>WARNING</b></font>{text[7:]}'
        elif "READY" in text:
            text = f'<font color="green"><b>READY</b></font>{text[5:]}'

        # Add a title and some text
        self.story.append(Paragraph(text))

    def add_image_to_story(self, image):
        # Load the image with PIL to get its size
        pil_image = PILImage.open(image)
        img_width_px, img_height_px = pil_image.size
        img_dpi = pil_image.info.get("dpi", (150,))[0]  # default DPI

        # 3. Convert pixels to points (1 inch = 72 points)
        img_width_pt = img_width_px * 72 / img_dpi
        img_height_pt = img_height_px * 72 / img_dpi
        aspect_ratio = img_height_pt / img_width_pt

        # 4. Set up PDF page and scale image to full page width minus margins
        page_width, page_height = A4
        left_margin = right_margin = 1 * inch
        available_width = page_width - left_margin - right_margin
        scaled_height = available_width * aspect_ratio
        self.story.append(Image(image, width=available_width, height=scaled_height))

    def create_full_story(self, path=None):
        # Create a PDF document
        timestamp = datetime.now().strftime("%y.%m.%d.%H.%M.%S")
        pdf_filename = f"reports/report.{timestamp}.pdf"
        self.doc = None
        self.doc = SimpleDocTemplate(pdf_filename, pagesize=A4)

        # Get some basic styles
        styles = getSampleStyleSheet()
        self.story = []

        # Connect console text
        try:
            self.main.console.signal_text_ready.connect(self.add_text_to_story)
        except:
            pass

        # Check path
        local_story = False
        if path is None:
            path = self.main.session["directory"] + "/mat"
            local_story = True

        # Create heading
        full_path = Path(path)
        id_path = full_path.parents[2].name  # Go up two levels from 'mat'
        self.add_text_to_story(f"<font color='black' size=16><b>Report for {id_path}</b></font>")

        files = self.get_sorted_mat_files(path)

        for file in files:
            if not local_story:
                self.story.append(Spacer(1, 12))
                self.add_text_to_story(f"<font color='blue' size=14><b>{file}</b></font>")
                self.story.append(Spacer(1, 12))
                dp.run_recon(raw_data_path=path + "/" + file, mode="Standalone", printer=self)
            else:
                raw_datas, info = self.main.history_list.getHistoryListInfo()
                if file in raw_datas:  # All  items unchecked
                    index = raw_datas.index(file)
                    self.story.append(Spacer(1, 12))
                    self.add_text_to_story(f"<font color='blue' size=14><b>{file}</b></font>")
                    self.add_text_to_story(f"<font color='black' size=12><b>{info[index]}</b></font>")
                    self.story.append(Spacer(1, 12))
                    dp.run_recon(raw_data_path=path + "/" + file, mode="Standalone", printer=self)

        try:
            self.main.console.signal_text_ready.disconnect(self.add_text_to_story)
        except:
            pass

        self.doc.build(self.story)

        print("READY: report created!")

    @staticmethod
    def get_sorted_mat_files(directory):
        """Return a list of .mat filenames in 'directory' sorted by creation time, excluding 'temp.mat'."""
        directory = Path(directory)
        mat_files = [
            f for f in directory.glob("*.mat")
            if f.name != "temp.mat"
        ]

        # Sort by creation time (oldest to newest)
        mat_files.sort(key=lambda f: f.stat().st_ctime)

        return [f.name for f in mat_files]