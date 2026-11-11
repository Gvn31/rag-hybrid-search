
import sys
from pathlib import Path


class RAGPipelineException(Exception):
    """Custom exception for the RAG Pipeline."""

    def __init__(self, error_message: str):
        # Store the original error message
        self.error_message = error_message

        try:
            # Get traceback information of the current exception
            _, _, exc_traceback = sys.exc_info()

            if exc_traceback is not None:
                # Extract filename and line number
                frame = exc_traceback.tb_frame
                self.filename = Path(frame.f_code.co_filename).name
                self.line_number = exc_traceback.tb_lineno

            else:
                # Default values when traceback is unavailable
                self.filename = "Unknown"
                self.line_number = 0

        except Exception:
            # Handle errors occurring during traceback extraction
            self.filename = "Unknown"
            self.line_number = 0

        # Initialize the parent Exception class
        super().__init__(self.__str__())

    def __str__(self):
        # Return formatted exception details
        return (
            f"Error: {self.error_message} | "
            f"File: {self.filename} | "
            f"Line: {self.line_number}"
        )
       
    