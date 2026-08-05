from datetime import datetime
import smtplib
from twilio.rest import Client

class AlertService:
    def __init__(self):
        self.client=Client()


    def send_alert(self, event, threat_score,snapshot_path=None):
        message=(
            f"CamShield Alert!\n"
            f"Event: {event}\n"
            f"Threat Score: {threat_score}"
        )

        self.client.messages.create(
            body=message,
            from_="+1XXXXXXXXXX",
            to="+91XXXXXXXXXX"
        )

        print("SMS sent successfully")
