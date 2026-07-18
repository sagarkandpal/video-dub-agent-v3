# YouTube dubbing ai agent

FastAPI-based service for processing YouTube videos and your videos with AI-powered transcription, translation, and summarization.

## Features
- Process any YouTube video
- Multi-language support (English, Hindi, Spanish, French, German, Chinese)
- AI-powered transcription using Whisper
- Automatic video summarization
- Translation capabilities
- Job queue management with status tracking
- RESTful API with Swagger documentation
- Async processing for better performance

## Tech Stack
- FastAPI - Web framework
- LangGraph - AI pipeline orchestration
- YouTube API - Video fetching
- Whisper - Speech-to-text
- Transformers - Summarization & translation
- Pytest - Testing
- Uvicorn - ASGI server

## Installation

### 1. Clone Repository
git clone https://github.com/yourusername/youtube-processor.git
cd youtube-processor

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Environment Variables
Create .env file:
YOUTUBE_API_KEY=your_youtube_api_key
DATABASE_URL=sqlite:///./jobs.db
LANGCHAIN_API_KEY=your_langchain_key

## Running the App

### Development
uvicorn app.main:app --reload --port 8000

### Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

### API Documentation
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

## API Endpoints

### Create Processing Job
POST /api/jobs
Request Body:
{
  "video_url": "https://youtube.com/watch?v=abc123",
  "language": "en"
}
Response:
{
  "job_id": "job-abc123",
  "status": "pending"
}

### Get Job Status
GET /api/jobs/{job_id}
Response:
{
  "id": "job-abc123",
  "status": "completed",
  "transcript": "Full video transcript...",
  "summary": "Video summary..."
}

### Health Check
GET /health
Response:
{
  "status": "healthy"
}

## Testing
Run all tests:
pytest

Run with coverage:
pytest --cov=app tests/

## Project Structure
youtube-processor/
├── app/
│   ├── main.py              # FastAPI app
│   ├── api/routes.py        # API endpoints
│   ├── core/
│   │   ├── job_manager.py   # Job management
│   │   └── helpers.py       # Utilities
│   ├── pipeline/
│   │   ├── graph.py         # LangGraph structure
│   │   └── nodes.py         # Pipeline nodes
│   └── models/schemas.py    # Pydantic models
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_api.py
│   ├── test_job_manager.py
│   ├── test_helpers.py
│   └── test_pipeline.py
├── requirements.txt
├── .env.example
└── README.md

## Docker Support
Build image:
docker build -t youtube-processor .

Run container:
docker run -p 8000:8000 --env-file .env youtube-processor

## Contributing
1. Fork the repo
2. Create feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## License
MIT License - see LICENSE file for details

## Author
Your Name
GitHub: @yourusername
LinkedIn: linkedin.com/in/yourprofile

Made with ❤️ and Python
