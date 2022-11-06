import os


def get_project_path():
    """to get the root path of the project"""
    pwd = os.getcwd()
    return os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
