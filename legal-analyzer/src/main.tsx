import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { BrowserRouter } from 'react-router-dom';
import LegalAIChat from './legalAI';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <LegalAIChat></LegalAIChat>
    </BrowserRouter>
  </StrictMode>,
)
