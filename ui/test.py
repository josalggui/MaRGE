from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create a PDF document
doc = SimpleDocTemplate("example.pdf", pagesize=A4)

# Get some basic styles
styles = getSampleStyleSheet()
story = []

# Add a title and some text
story.append(Paragraph("My First PDF", styles['Title']))
story.append(Spacer(1, 12))
story.append(Paragraph("This PDF was created using Python and ReportLab!", styles['Normal']))

# Build the PDF
doc.build(story)