# coding=utf-8
"""log util"""
import logbook
from logbook.compat import redirect_logging
from utils.file_util import ensure_dir_exists


class Logger(object):
    def __init__(self, date_format='%Y-%m-%d', level=logbook.DEBUG,
                 format_string='{record.time:%Y-%m-%d %H:%M:%S}|{record.level_name}|{record.message}'):
        self.format_string = format_string
        self.date_format = date_format
        self.level = level
        # all logging calls that happen after this call will transparently be redirected to Logbook
        redirect_logging()

    def set_stream_handler(self, level=None, format_string=None, bubble=True):
        level = level or self.level
        format_string = format_string or self.format_string
        handler = logbook.StderrHandler(level=level, bubble=bubble)
        handler.formatter.format_string = format_string
        handler.push_application()

    def set_time_rotating_file_handler(self, file_name, date_format=None, level=None, format_string=None, bubble=True,
                                       backup_count=10):
        ensure_dir_exists(file_name, is_dir=False)
        level = level or self.level
        date_format = date_format or self.date_format
        format_string = format_string or self.format_string
        handler = logbook.TimedRotatingFileHandler(file_name, level=level, bubble=bubble, date_format=date_format,
                                                   backup_count=backup_count)
        handler.formatter.format_string = format_string
        handler.push_application()
