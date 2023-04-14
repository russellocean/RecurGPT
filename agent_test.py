import unittest
from unittest.mock import patch
from agent_tools import ListFilesAndDirectoriesTool

class TestListFilesAndDirectoriesTool(unittest.TestCase):
    
    def setUp(self):
        self.tool = ListFilesAndDirectoriesTool()
    
    def test_valid_directory(self):
        path = "/Users/russellocean/Dev/RecurGPT"
        expected_directories = "Directories: .chroma, __pycache__, .git"
        expected_files = "Files: view_directory.py, config.py, agent_utils.py, prompt.txt, custom_parser.py, agent_test.py, .gitignore, utils.py, agent.py, main.py, agent_components.py, project_loader.py, file_viewer.py, agent_tools.py"
        actual_output = self.tool._run(path)
        self.assertIn(expected_directories, actual_output)
        self.assertIn(expected_files, actual_output)
        
    def test_invalid_directory(self):
        path = "/path/to/invalid/directory"
        expected_output = "Error: The specified path '/path/to/invalid/directory' does not exist. Please provide a valid directory."
        actual_output = self.tool._run(path)
        self.assertEqual(actual_output, expected_output)
        
    def test_non_directory_path(self):
        path = "/Users/russellocean/Dev/RecurGPT/main.py"
        expected_output = "Error: The specified path '/Users/russellocean/Dev/RecurGPT/main.py' is not a directory. Please provide a valid directory."
        actual_output = self.tool._run(path)
        self.assertEqual(actual_output, expected_output)
        
    def test_unexpected_error(self):
        path = "/Users/russellocean/Dev/RecurGPT"

        # Define a function to raise TypeError when called
        def raise_error(*args):
            raise TypeError("Invalid argument")

        # Patch os.listdir with the raise_error function
        with patch('os.listdir', side_effect=raise_error):
            expected_output = "Error: An internal function was called with an invalid argument."
            actual_output = self.tool._run(path)
            self.assertEqual(actual_output, expected_output)

    
    # def test_async_not_implemented(self):
    #     with self.assertRaises(NotImplementedError):
    #         self.tool._arun()
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestListFilesAndDirectoriesTool)
    unittest.TextTestRunner(verbosity=2).run(suite)