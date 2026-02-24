from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import json

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

import db
from dependencies import settings

logger = logging.getLogger(__name__)

class CalendarService:
    def __init__(self, user_id_hash: str):
        self.user_id_hash = user_id_hash
        self.service = None

    async def _get_credentials(self) -> Optional[Credentials]:
        """Retrieve and refresh user credentials."""
        tokens = await db.get_oauth_tokens(self.user_id_hash, "google")
        if not tokens:
            logger.warning(f"No Google tokens found for user {self.user_id_hash}")
            return None

        creds = Credentials(
            token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=tokens.get('scope', '').split()
        )

        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Update tokens in DB
                await db.store_oauth_tokens(self.user_id_hash, "google", {
                    'access_token': creds.token,
                    'refresh_token': creds.refresh_token, # Might remain same
                    'expiry': creds.expiry,
                    'scope': ' '.join(creds.scopes) if creds.scopes else tokens.get('scope')
                })
            except Exception as e:
                logger.error(f"Failed to refresh token for user {self.user_id_hash}: {e}")
                return None
        
        return creds

    async def get_service(self):
        """Get the Calendar API service instance."""
        if self.service:
            return self.service
            
        creds = await self._get_credentials()
        if not creds:
            return None
            
        try:
            self.service = build('calendar', 'v3', credentials=creds)
            return self.service
        except Exception as e:
            logger.error(f"Failed to build calendar service: {e}")
            return None

    async def list_events(self, max_results: int = 10) -> List[Dict]:
        """List upcoming events."""
        service = await self.get_service()
        if not service:
            return []

        try:
            now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            events_result = service.events().list(
                calendarId='primary', timeMin=now,
                maxResults=max_results, singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            return events
        except Exception as e:
            logger.error(f"Calendar list error: {e}")
            return []

    async def create_event(self, summary: str, start_time: datetime, end_time: datetime, description: str = "") -> Optional[str]:
        """Create a new event."""
        service = await self.get_service()
        if not service:
            return None

        event = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'UTC',
            },
        }

        try:
            event = service.events().insert(calendarId='primary', body=event).execute()
            logger.info(f"Event created: {event.get('htmlLink')}")
            return event.get('id')
        except Exception as e:
            logger.error(f"Calendar create error: {e}")
            return None
