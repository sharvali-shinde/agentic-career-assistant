"""Application configuration and environment settings."""
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Model
OPENAI_MODEL = os.getenv("MODEL_NAME")

# Profile
NAME = os.getenv("NAME")
LINKEDIN_URL =  os.getenv("LINKEDIN_URL")
CAREER_INFO_PDF_PATH = os.getenv("CAREER_INFO_PDF_PATH")
SUMMARY_PATH = os.getenv("SUMMARY_PATH")

# Pushover
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")