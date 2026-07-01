from fpdf import FPDF
from io import BytesIO
from datetime import datetime

class SatyaLensPDF(FPDF):
    def header(self):
        # Header title
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(37, 99, 235) # Primary Blue (blue-600)
        self.cell(0, 10, 'SATYALENS SECURITY ANALYSIS', ln=True, align='L')
        # Subtitle
        self.set_font('helvetica', 'I', 9)
        self.set_text_color(107, 114, 128) # Secondary Gray (gray-500)
        self.cell(0, 5, 'AI-Based Deepfake & Synthetic Identity-Risk Assessment', ln=True, align='L')
        
        # Horizontal divider rule
        self.set_draw_color(229, 231, 235)
        self.line(10, 26, 200, 26)
        self.ln(8)
        
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(156, 163, 175)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Research Prototype | Non-Automated Decision Support', 0, 0, 'C')

def generate_pdf_report(scan_results, media_type):
    """
    Generates a PDF report containing deepfake scan scores, risk labels, liveness breakdowns,
    and recommended security procedures.
    Returns: Bytes of the PDF file.
    """
    pdf = SatyaLensPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Metadata Block
    pdf.set_font('helvetica', 'B', 11)
    pdf.set_text_color(17, 24, 39) # Slate 900
    pdf.cell(0, 7, 'SCAN METADATA', ln=True)
    
    pdf.set_font('helvetica', '', 9.5)
    pdf.set_text_color(55, 65, 81) # Slate 700
    pdf.cell(50, 6, 'Date & Time:', 0)
    pdf.cell(0, 6, datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC+05:30'), ln=True)
    pdf.cell(50, 6, 'Media Filename:', 0)
    pdf.cell(0, 6, str(scan_results.get('filename', 'Unknown')), ln=True)
    pdf.cell(50, 6, 'Analysis Type:', 0)
    pdf.cell(0, 6, f'{media_type.capitalize()} Deepfake & Liveness Scan', ln=True)
    pdf.ln(5)
    
    # Primary Verdict Card (Colored Rectangle Box)
    pdf.set_fill_color(248, 250, 252) # Light slate background (slate-50)
    pdf.rect(10, 52, 190, 42, 'F')
    pdf.set_xy(12, 54)
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 6, 'THREAT CLASS VERDICT', ln=True)
    
    # Risk assessment styling based on label
    risk_lbl = scan_results.get('risk', 'Low Risk')
    pdf.set_x(12)
    pdf.set_font('helvetica', 'B', 11)
    if risk_lbl == 'Low Risk':
        pdf.set_text_color(22, 163, 74) # Green-600
    elif risk_lbl == 'Suspicious':
        pdf.set_text_color(245, 158, 11) # Amber-500
    else:
        pdf.set_text_color(220, 38, 38) # Red-600
    pdf.cell(50, 6, 'Threat Assessment:', 0)
    pdf.cell(0, 6, risk_lbl.upper(), ln=True)
    
    # Classification and synthetic score
    pdf.set_x(12)
    pdf.set_text_color(55, 65, 81)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(50, 6, 'Synthetic Probability:', 0)
    pdf.cell(0, 6, f'{scan_results.get("synthetic_probability", 0.0):.2%}', ln=True)
    
    # Liveness score (for video)
    pdf.set_x(12)
    if 'liveness_score' in scan_results:
        pdf.cell(50, 6, 'Passive Liveness Score:', 0)
        pdf.cell(0, 6, f'{scan_results.get("liveness_score")}/100 ({scan_results.get("liveness_label")})', ln=True)
    else:
        pdf.cell(50, 6, 'Liveness Proof:', 0)
        pdf.cell(0, 6, 'N/A (Image-only analysis limits liveness security)', ln=True)
        
    pdf.set_xy(10, 98)
    pdf.ln(5)
    
    # Recommended Action Card
    pdf.set_font('helvetica', 'B', 11)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 7, 'RECOMMENDED SECURITY ACTIONS', ln=True)
    
    pdf.set_font('helvetica', '', 9.5)
    action_text = scan_results.get('action', '')
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(229, 231, 235)
    
    pdf.multi_cell(0, 5, action_text, border=1, fill=True)
    pdf.ln(6)
    
    # Detailed Component Scores (Liveness)
    if 'components' in scan_results:
        pdf.set_font('helvetica', 'B', 11)
        pdf.set_text_color(17, 24, 39)
        pdf.cell(0, 7, 'DETAILED METRIC COMPONENT BREAKDOWN', ln=True)
        
        pdf.set_font('helvetica', 'B', 9.5)
        pdf.set_fill_color(243, 244, 246)
        pdf.set_text_color(55, 65, 81)
        pdf.cell(95, 7, ' Metric / Component', border=1, fill=True)
        pdf.cell(95, 7, ' Score / Value', border=1, fill=True, ln=True)
        
        pdf.set_font('helvetica', '', 9)
        components = scan_results['components']
        for key, val in components.items():
            nice_key = key.replace('_', ' ').capitalize()
            pdf.cell(95, 6, f'  {nice_key}', border=1)
            pdf.cell(95, 6, f'  {val}%', border=1, ln=True)
        pdf.ln(6)
        
    # Aggregate stats (Video frames)
    if 'video_metrics' in scan_results:
        pdf.set_font('helvetica', 'B', 11)
        pdf.set_text_color(17, 24, 39)
        pdf.cell(0, 7, 'VIDEO TIMELINE AGGREGATION DETAIL', ln=True)
        
        pdf.set_font('helvetica', 'B', 9.5)
        pdf.set_fill_color(243, 244, 246)
        pdf.set_text_color(55, 65, 81)
        pdf.cell(95, 7, ' Aggregation Factor', border=1, fill=True)
        pdf.cell(95, 7, ' Score', border=1, fill=True, ln=True)
        
        pdf.set_font('helvetica', '', 9)
        v_metrics = scan_results['video_metrics']
        for key, val in v_metrics.items():
            nice_key = key.replace('_', ' ').capitalize()
            # Check if float to format as percentage
            str_val = f'{val:.2%}' if isinstance(val, float) and ('ratio' in key or 'probability' in key or 'risk' in key) else str(val)
            pdf.cell(95, 6, f'  {nice_key}', border=1)
            pdf.cell(95, 6, f'  {str_val}', border=1, ln=True)
        pdf.ln(6)

    # Disclaimer Section
    pdf.set_font('helvetica', 'B', 9.5)
    pdf.set_text_color(220, 38, 38) # Red alert (red-600)
    pdf.cell(0, 6, 'RESPONSIBLE AI USE & PROTOCOL DISCLAIMER', ln=True)
    
    pdf.set_font('helvetica', '', 8)
    pdf.set_text_color(107, 114, 128)
    disclaimer_body = (
        "1. SatyaLens is a research prototype. It estimates threat classes based on a specific model card and visual heuristics. "
        "It does not guarantee 100% detection rates and is susceptible to false results under low-light or compressed media.\n"
        "2. Do not use this tool for automated approvals or rejections in production KYC onboarding, vetting pipelines, or legal casework.\n"
        "3. Results must always be escalated to a trained human investigator for secondary verification. Biometric spoof detections "
        "calculated here are non-certified heuristics."
    )
    pdf.multi_cell(0, 4.5, disclaimer_body)
    
    # Return as bytes
    pdf_bytes = pdf.output()
    
    # Depending on fpdf version output might return str or bytes
    if isinstance(pdf_bytes, str):
        return pdf_bytes.encode('latin1')
    return pdf_bytes
