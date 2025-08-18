import os

def test_dirs_exist():
    assert os.path.isdir("code")
    assert os.path.isdir("data")
    assert os.path.isdir("tests")
