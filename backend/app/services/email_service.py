import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, Optional

def send_risk_alert_email(
    recipient_email: str,
    assessment_id: str,
    risk_level: str,
    probability: float,
    feature_importances: list,
    patient_name: str = "Anonymous Patient"
) -> bool:
    """
    Sends a real-time risk alert email.
    In a production setting, uses SMTP_USERNAME and SMTP_PASSWORD from env vars.
    If not set, it simulates the email send by printing to console.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SMTP_SENDER", smtp_username or "cogai.alerts@example.com")

    confidence_pct = probability * 100

    # Build Top Factors HTML
    factors_html = "<ul>"
    for f in feature_importances[:3]:
        name = f.get("feature", "Unknown")
        contribution = f.get("contribution", 0) * 100
        direction = "increased" if f.get("direction") == "positive" else "reduced"
        factors_html += f"<li><b>{name}</b> ({contribution:.1f}% impact) - {direction} risk</li>"
    factors_html += "</ul>"

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
          <h2 style="color: #D32F2F;">⚠️ Urgent: High Cognitive Risk Detected</h2>
          <p>This is an automated alert from the Cog-AI Prediction Engine.</p>
          <hr/>
          <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <tr style="background-color: #f9f9f9;">
              <td style="padding: 10px; border-bottom: 1px solid #ddd;"><b>Patient:</b></td>
              <td style="padding: 10px; border-bottom: 1px solid #ddd;">{patient_name}</td>
            </tr>
            <tr style="background-color: #fff;">
              <td style="padding: 10px; border-bottom: 1px solid #ddd;"><b>Assessment ID:</b></td>
              <td style="padding: 10px; border-bottom: 1px solid #ddd;">{assessment_id}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
              <td style="padding: 10px; border-bottom: 1px solid #ddd;"><b>Risk Level:</b></td>
              <td style="padding: 10px; border-bottom: 1px solid #ddd; color: #D32F2F; font-weight: bold;">{risk_level}</td>
            </tr>
            <tr style="background-color: #fff;">
              <td style="padding: 10px; border-bottom: 1px solid #ddd;"><b>AI Confidence:</b></td>
              <td style="padding: 10px; border-bottom: 1px solid #ddd;">{confidence_pct:.1f}%</td>
            </tr>
          </table>
          
          <h3>Top Contributing Factors</h3>
          {factors_html}
          
          <div style="background-color: #FFF3E0; padding: 15px; border-left: 4px solid #FF9800; margin-top: 20px;">
            <strong>Immediate Action Recommended:</strong> Please contact the patient to schedule a comprehensive clinical neurological evaluation.
          </div>
        </div>
      </body>
    </html>
    """

    # If no SMTP credentials provided, simulate successful send (For Hackathon/Testing)
    if not smtp_username or not smtp_password:
        print("\n" + "="*50)
        print(f"📧 [SIMULATED EMAIL SENT TO {recipient_email}]")
        print("Set SMTP_USERNAME and SMTP_PASSWORD in .env to send real emails.")
        print(f"Subject: Cog-AI Alert: High Risk Detected ({confidence_pct:.1f}%)")
        print("Body (HTML formatted omitted for brevity) - Check factors:")
        for f in feature_importances[:3]:
            print(f"  - {f.get('feature')}: {f.get('contribution', 0)*100:.1f}%")
        print("="*50 + "\n")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Cog-AI Alert: High Risk Detected ({confidence_pct:.1f}%)"
        msg["From"] = sender_email
        msg["To"] = recipient_email

        # Attach HTML content
        part = MIMEText(html_content, "html")
        msg.attach(part)

        # Send Email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"[EmailService] Real email successfully sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"[EmailService] Failed to send email: {e}")
        return False
