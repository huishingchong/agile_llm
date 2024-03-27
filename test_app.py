import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
from dotenv import load_dotenv
import os
from app import create_jobs_csv, get_job, clean_html, query_job_listings
import csv

class TestApp(unittest.TestCase):
    def setUp(self):
        env_path = Path('.') / '.env'
        load_dotenv(dotenv_path=env_path)
        self.reed_key = os.getenv('REED_API_KEY')
    
    def test_create_jobs_csv_creates_csv_rows_with_expected_data(self):
        job_listings = query_job_listings("Software Engineer", "london", self.reed_key)
        create_jobs_csv(job_listings, self.reed_key)

        # Check if the CSV file is created
        self.assertTrue(os.path.exists('job_listings.csv'))

        # Check if the CSV file contains the expected datatypes
        with open('job_listings.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.assertIsInstance(row['Job Title'], str)
                self.assertIsInstance(row['Location'], str)
                self.assertTrue(row['Part-time'] in ['True', 'False'])
                self.assertTrue(row['Full-time'] in ['True', 'False'])
    
    def test_csv_file_not_empty_after_successfully_writing_with_job_listings(self):        
        self.assertTrue(os.path.exists('job_listings.csv'))
        # Check if the CSV file contains rows
        with open('job_listings.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            row_count = sum(1 for row in reader)
            self.assertGreater(row_count, 0, "CSV file should contain at least one row")


    @patch('app.get')
    def test_valid_query_returns_job_listings(self, mocked_get):
        # Mocked response for valid query
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.json.return_value = {"results": [{"jobId": 1, "jobTitle": "Software Engineer"}]}

        # Call the function with valid arguments
        job_listings = query_job_listings("Software Engineer", "London", self.reed_key)

        self.assertTrue(job_listings)
        self.assertTrue(len(job_listings) > 0)

    @patch('app.get')
    def test_invalid_query_returns_empty_job_listings(self, mocked_get):
        # Mocked response for invalid query
        mocked_get.return_value.status_code = 404

        # Call the function with valid arguments
        job_listings = query_job_listings("Invalid Job", "Invalid Location", "Wrong key")
        self.assertFalse(job_listings)


    def test_get_job_successfully_extracts_correct_keyword_from_query(self):
        # Test that LLM is returning correct keyword from query
        job_title = get_job("What are the responsibilities of a data scientist?")
        self.assertIn("data scientist", job_title)

    def test_get_job_successfully_extracts_correct_keyword_from_another_query(self):
        # Test that LLM is returning correct keyword from query
        job_title = get_job("Tell me about the emerging skills in computer vision industry.")
        self.assertIn("computer vision", job_title)

    def test_clean_html_get_rids_of_tags_from_input_text(self):
        # Test if clean_html helper function removes html tags properly
        html_text = "<p>This is a <b>test</b> text to check the cleanhtml function!</p>"
        cleaned_text = clean_html(html_text)
        self.assertEqual(cleaned_text, "This is a test text to check the cleanhtml function!")
    
    def test_clean_html_with_already_cleaned_input_text_returns_input_text(self):
        html_text = "This is a perfectly clean text!"
        cleaned_text = clean_html(html_text)
        self.assertEqual(cleaned_text, "This is a perfectly clean text!")

if __name__ == '__main__':
    unittest.main()