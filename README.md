# Legal AI Assistant 

### Setup and running

- clone the repository `https://github.com/veeramani27/AI_legal_assistant`
- cd AI_legal_assistant
- using uv or pip install all the packages in the requirements.txt
    - UV
        - uv init
        - uv add -r requirements.txt
    - pip 
        - pip install -r requirements.txt

- open three terminals in PWD (present working directory)

- Langgraph Backend
    - cd langgraph_legal_ai
    - uvicorn legal_agent_wrapper:api --port 8787 --reload

- Backend API server
    - cd fastapi_server
    - uvicorn fastapi_legal:api --reload

- Frontend react 
    - cd legal-analyser
    - npm install
    - npm run dev

Open the browser and go to `localhost:5173`