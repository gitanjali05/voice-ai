# Voice-ai

A Python-based Voice AI assistant that combines speech recognition, language model APIs, and a web chat interface to enable real-time voice and text conversations.

📁 Project Structure

voice-ai/

├── main.py           # Entry point for the voice assistant

├── main_data.py      # Entry point with data-augmented capabilities

├── api.py            # Core API integration (LLM / speech services)

├── api_data.py       # API integration with data tool support

├── data_tools.py     # Tools for querying and processing data

├── web_chat.py       # Web-based chat interface

├── voice/            # Voice assets and audio utilities

└── data/             # Data files used by the assistant

🚀 Features

🎤 Voice input and speech recognition
🤖 AI-powered responses via LLM API integration
💬 Web-based chat interface (web_chat.py)
📊 Data querying and tool use capabilities
🔊 Voice output / text-to-speech support


🛠️ Requirements

Python 3.8+
Required packages (install via pip):

bashpip install -r requirements.txt

If no requirements.txt is present, common dependencies include:
openai, speechrecognition, pyaudio, flask, requests


▶️ Getting Started
1. Clone the repository
bashgit clone https://github.com/gitanjali05/voice-ai.git
cd voice-ai
2. Install dependencies
bashpip install -r requirements.txt
3. Set up environment variables
Create a .env file in the root directory and add your API keys:
envOPENAI_API_KEY=your_openai_api_key
# Add any other required keys
4. Run the voice assistant
bashpython main.py
5. Run the web chat interface
bashpython web_chat.py
Then open your browser and go to http://localhost:5000 (or whichever port is configured).

📊 Data Mode
To run the assistant with data tool capabilities:
bashpython main_data.py
This enables the assistant to query and reason over structured data using the tools defined in data_tools.py.

📝 Notes

Make sure your microphone is connected and accessible when running the voice mode.
The voice/ directory contains audio files or voice model assets used during runtime.
The data/ directory contains datasets or knowledge files referenced by the assistant.


🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

📄 License
This project is open source. See LICENSE for details.

Built with Python 🐍 and powered by AI
