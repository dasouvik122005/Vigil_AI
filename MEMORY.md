# Project Memory

This file serves as a persistent memory and progress tracker for the STAMPER_TSLR project.

## Current State
- **Project Goal**: Build an AI-powered Adaptive Decision Intelligence Platform (Hackathon Track 1).
- **Repository**: Connected to `https://github.com/GITtridib22/STAMPER_TSLR.git`
- **Status**: The MVP prototype is built and running locally. Both FastAPI backend and Vite/React frontend are fully functional. README, requirements.txt, and setup.ps1 have been added for easier deployment. A `vercel.json` file has been added for deploying the entire monorepo to Vercel.

## Next Steps
- Implement a more complex machine learning model in Python to replace the mock confidence scoring.
- Allow users to inject actual datasets instead of random simulations.
- Fine-tune CSS layout based on real-world constraints.

## Important Context & Decisions
- No specific ML framework decided yet.
- Hard mode: Must gracefully handle 20-30% missing or corrupted data.
- Core Challenge: The AI must know when to not trust itself and trigger a human-in-the-loop fallback.
