# Bolo — AI Doubt Solver in Regional Languages

An AI-powered doubt-solving tool that explains student questions in Hindi,
Hinglish, or English, and tracks which subjects/languages are most asked
for teacher-facing analytics.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get a free Groq API key: https://console.groq.com -> API Keys

3. Set your key as an environment variable:
   ```
   # Mac/Linux
   export GROQ_API_KEY="your-key-here"

   # Windows (PowerShell)
   $env:GROQ_API_KEY="your-key-here"
   ```

   OR create a `.streamlit/secrets.toml` file with:
   ```
   GROQ_API_KEY = "your-key-here"
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

## Deploying (for your Live Demo link)

1. Push this project to a GitHub repo.
2. Go to https://share.streamlit.io, connect your GitHub repo.
3. Add `GROQ_API_KEY` under the app's "Secrets" settings.
4. Deploy — you'll get a public URL to submit.
