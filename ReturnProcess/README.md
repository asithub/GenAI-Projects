### Create a fresh virtual environment
`python -m venv venv`

### Activate virtual environment
`.\venv\Scripts\Activate.ps1`

### To install the packages from your text file
`pip install -r requirements.txt`

### Start FastAPI
`python -m uvicorn app.main:app --reload`

Uvicorn running on http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs