import unittest
from unittest import mock
import tempfile
import os
import pkg_resources
import json
from nullcal.tools.create_injection import main

class TestCreateInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary directory for testing."""
        cls.test_dir = tempfile.mkdtemp()
        cls.ET1_TEST_frame_path = os.path.join(cls.test_dir, 'ET1-TEST-2024-16.gwf')
        cls.ET2_TEST_frame_path = os.path.join(cls.test_dir, 'ET2-TEST-2024-16.gwf')
        cls.ET3_TEST_frame_path = os.path.join(cls.test_dir, 'ET3-TEST-2024-16.gwf')
        cls.ET1_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, 'ET1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf')
        cls.ET2_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, 'ET2-TEST_WITH_SIGNAL_FRAME-2024-16.gwf')
        cls.ET3_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, 'ET3-TEST_WITH_SIGNAL_FRAME-2024-16.gwf')
        cls.ET1_TEST_PSD_frame_path = os.path.join(cls.test_dir, 'ET1-TEST_PSD-2024-16.gwf')
        cls.ET2_TEST_PSD_frame_path = os.path.join(cls.test_dir, 'ET2-TEST_PSD-2024-16.gwf')
        cls.ET3_TEST_PSD_frame_path = os.path.join(cls.test_dir, 'ET3-TEST_PSD-2024-16.gwf')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.ET1_TEST_frame_path):
            os.remove(cls.ET1_TEST_frame_path)
        if os.path.exists(cls.ET2_TEST_frame_path):
            os.remove(cls.ET2_TEST_frame_path)
        if os.path.exists(cls.ET3_TEST_frame_path):
            os.remove(cls.ET3_TEST_frame_path)   
        if os.path.exists(cls.ET1_TEST_WITH_SIGNAL_FRAME_frame_path):
            os.remove(cls.ET1_TEST_WITH_SIGNAL_FRAME_frame_path)
        if os.path.exists(cls.ET2_TEST_WITH_SIGNAL_FRAME_frame_path):
            os.remove(cls.ET2_TEST_WITH_SIGNAL_FRAME_frame_path)
        if os.path.exists(cls.ET3_TEST_WITH_SIGNAL_FRAME_frame_path):
            os.remove(cls.ET3_TEST_WITH_SIGNAL_FRAME_frame_path)
        if os.path.exists(cls.ET1_TEST_PSD_frame_path):
            os.remove(cls.ET1_TEST_PSD_frame_path)
        if os.path.exists(cls.ET2_TEST_PSD_frame_path):
            os.remove(cls.ET2_TEST_PSD_frame_path)
        if os.path.exists(cls.ET3_TEST_PSD_frame_path):
            os.remove(cls.ET3_TEST_PSD_frame_path)            

    def test_generate_config(self):
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        with mock.patch('sys.argv', ['nullcal-create-injection', '--generate-config', config_file_path]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Check that the config file was generated
        self.assertTrue(os.path.exists(config_file_path))

        # Verify the contents of the generated config file
        with open(config_file_path, 'r') as f:
            generated_content = f.read()

        # Load the default config file
        default_config_file_path = pkg_resources.resource_filename('nullcal.tools', 'default_config_create_injection.ini')
        with open(default_config_file_path, 'r') as f:
            default_generated_content = f.read()

        # Clean up the temporary config file
        os.remove(config_file_path)

        # Compare the content of both files
        self.assertEqual(generated_content.strip(), default_generated_content.strip())

    def test_create_injection(self):
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename('nullcal.tools', 'example_signal_parameters_create_injection.json')
        with mock.patch('sys.argv', ['nullcal-create-injection',
                                     '--generate-config', config_file_path,
                                     '--signal-parameters', example_signal_parameters_create_injection_path,
                                     '--outdir', self.test_dir,
                                     '--label', 'TEST',
                                     '--start-time', '2024',
                                     '--duration', '16',
                                     '--detectors', 'ET']):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch('sys.argv', ['nullcal-create-injection', '--config', config_file_path]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Clean up the temporary config file
        os.remove(config_file_path)

        # Check that the output file is created        
        self.assertTrue(os.path.exists(self.ET1_TEST_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.ET2_TEST_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.ET3_TEST_frame_path), "Output file was not created.")

    def test_create_injection_with_signal_frame(self):
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename('nullcal.tools', 'example_signal_parameters_create_injection.json')
        with mock.patch('sys.argv', ['nullcal-create-injection',
                                     '--generate-config', config_file_path,
                                     '--signal-parameters', example_signal_parameters_create_injection_path,
                                     '--outdir', self.test_dir,
                                     '--label', 'TEST_WITH_SIGNAL_FRAME',
                                     '--start-time', '2024',
                                     '--duration', '16',
                                     '--detectors', 'ET',
                                     '--signal-files', json.dumps(f'{{"ET1": "{self.ET1_TEST_frame_path}", "ET2": "{self.ET2_TEST_frame_path}", "ET3": "{self.ET3_TEST_frame_path}"}}'),
                                     '--signal-file-channels', json.dumps('{"ET1": "ET1:STRAIN", "ET2": "ET2:STRAIN", "ET3": "ET3:STRAIN"}')]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch('sys.argv', ['nullcal-create-injection', '--config', config_file_path]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Clean up the temporary config file
        os.remove(config_file_path)

        # Check that the output file is created
        self.assertTrue(os.path.exists(self.ET1_TEST_WITH_SIGNAL_FRAME_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.ET2_TEST_WITH_SIGNAL_FRAME_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.ET3_TEST_WITH_SIGNAL_FRAME_frame_path), "Output file was not created.")

    def test_create_injection_with_custom_psds(self):
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename('nullcal.tools', 'example_signal_parameters_create_injection.json')
        current_dir = os.path.dirname(__file__)
        mock_psd_path = os.path.join(current_dir, 'mock_psd.txt')
        with mock.patch('sys.argv', ['nullcal-create-injection',
                                     '--generate-config', config_file_path,
                                     '--signal-parameters', example_signal_parameters_create_injection_path,
                                     '--outdir', self.test_dir,
                                     '--label', 'TEST_PSD',
                                     '--start-time', '2024',
                                     '--duration', '16',
                                     '--detectors', 'ET',
                                     '--psds', json.dumps(f'{{"ET1": "{mock_psd_path}", "ET2": "{mock_psd_path}", "ET3": "{mock_psd_path}"}}')]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch('sys.argv', ['nullcal-create-injection', '--config', config_file_path]):
            try:                
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Clean up the temporary config file
        #os.remove(config_file_path)

        # Check that the output file is created        
        self.assertTrue(os.path.exists(self.ET1_TEST_PSD_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.ET2_TEST_PSD_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.ET3_TEST_PSD_frame_path), "Output file was not created.")        

if __name__ == '__main__':
    unittest.main()