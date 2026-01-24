# Legal AI Assistant 

### Setup and running

- `git clone https://github.com/veeramani27/AI_legal_assistant`
- `cd AI_legal_assistant`
- create a venv using 
    - `python -m venv .venv`
    or 
    - `uv venv`
- `.venv/Script/activate`

- using uv or pip install all the packages in the requirements.txt
    - UV
        - `uv add -r requirements.txt`
    - pip 
        - `pip install -r requirements.txt`

- copy .env.sample file and fill the api keys
- open three terminals in PWD (present working directory)

- TERMINAL 1 : Langgraph Backend
    - `cd langgraph_legal_ai`
    - `uvicorn legal_agent_wrapper:api --port 8787 --reload`

- TERMINAL 2 : Backend API server
    - `cd fastapi_server`
    - `uvicorn fastapi_legal:api --reload`

- TERMINAL 3 : Frontend react 
    - `cd legal-analyzer`
    - `npm install`
    - `npm run dev`

Open the browser and go to `http://localhost:5173`