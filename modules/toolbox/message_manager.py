from datetime import datetime
from typing import List
import queue
import threading


class MessageManager:
    def __init__(self, max_messages: int = 100):
        self._messages: List[str] = []
        self._max_messages = max_messages
        self._message_queue = queue.Queue()
        self._lock = threading.Lock()

        # ANSI-style formatting for different message types
        self._formats = {
            "INFO": "ℹ️",  # Info icon
            "SUCCESS": "✅",  # Checkmark
            "WARNING": "⚠️",  # Warning icon
            "ERROR": "❌",  # Error icon
        }

    def add_message(self, message: str, message_type: str = "INFO") -> None:
        """Add a new message with minimal timestamp and icon."""
        # Only show hours:minutes for timestamps
        timestamp = datetime.now().strftime("%H:%M")
        icon = self._formats.get(message_type, "•")

        # Format filename paths to be more readable
        if "Processing file" in message or "Created batch folder" in message:
            message = self._format_path(message)

        formatted_message = f"{icon} {message}"

        with self._lock:
            self._messages.append(formatted_message)
            if len(self._messages) > self._max_messages:
                self._messages.pop(0)

    def _format_path(self, message: str) -> str:
        """Format file paths to be more concise and readable."""
        if "GRADIO_TEMP_DIR" in message:
            # Extract just the filename from temp path
            filename = message.split("\\")[-1]
            return message.split(":")[0] + ": " + filename
        elif "batch_" in message:
            # Shorten batch folder path
            return message.replace("../outputs/", "")
        return message

    def add_success(self, message: str) -> None:
        """Add a success message."""
        self.add_message(message, "SUCCESS")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.add_message(message, "WARNING")

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.add_message(message, "ERROR")

    def get_messages(self) -> str:
        """Get all messages as a single string with spacing between different types."""
        with self._lock:
            # Add a blank line between different message types for readability
            formatted = []
            last_type = None
            for msg in self._messages:
                current_type = next(
                    (t for t in self._formats if self._formats[t] in msg), None
                )
                if last_type and current_type != last_type:
                    formatted.append("")  # Add spacing between different types
                formatted.append(msg)
                last_type = current_type
            return "\n".join(formatted)

    def clear(self) -> None:
        """Clear all messages."""
        with self._lock:
            self._messages.clear()
