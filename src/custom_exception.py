import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail=sys):
        super().__init__(error_message)
        self.error_message = self.get_detail_error_message(error_message, error_detail)

    @staticmethod
    def get_detail_error_message(error_message, error_detail=sys):
        _, _, exc_traceback = error_detail.exc_info()
        file_name = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        return f"Error occurred in file {file_name}, line {line_number}:\nMessage: {error_message}"

    def __str__(self):
        return self.error_message


