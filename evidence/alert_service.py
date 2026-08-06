import os
import logging
from datetime import datetime
from twilio.rest import Client

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self, alert_threshold: float = 80.0):
        self.alert_threshold = alert_threshold
        self.alert_active = False
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "+1XXXXXXXXXX")
        self.to_number = os.getenv("TWILIO_TO_NUMBER", "+91XXXXXXXXXX")
        self.client = None

        if self.account_sid and self.auth_token:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Twilio Client initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Twilio Client: {e}")
        else:
            logger.info("Twilio credentials missing. SMS alerts will run in simulation mode.")

    def process_threat(self, threat_score: float, event_name: str, snapshot_path: str = None) -> bool:
        """
        Process threat score with deduplication logic.
        Triggers SMS only once when threat_score >= threshold.
        Resets trigger when threat_score drops below threshold.
        """
        if threat_score >= self.alert_threshold:
            if not self.alert_active:
                sent = self.send_alert(event_name, threat_score, snapshot_path)
                self.alert_active = True
                return sent
        else:
            if self.alert_active:
                logger.info(f"Threat score dropped to {threat_score}%. Resetting SMS alert trigger.")
                self.alert_active = False

        return False

    def send_alert(self, event: str, threat_score: float, snapshot_path: str = None) -> bool:
        time_str = datetime.now().strftime("%H:%M:%S")
        message_body = (
            f"ALERT!\n"
            f"Threat Score: {int(threat_score)}%\n"
            f"Incident: {event}\n"
            f"Time: {time_str}\n"
            f"Please check immediately."
        )

        logger.warning(f"[SMS TRIGGERED] {message_body}")

        if self.client:
            try:
                self.client.messages.create(
                    body=message_body,
                    from_=self.from_number,
                    to=self.to_number
                )
                logger.info("Twilio SMS sent successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to send Twilio SMS: {e}")
                return False
        else:
            logger.info("Twilio SMS simulated (Twilio credentials not configured).")
            return False

