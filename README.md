# Final-Year-Project(house plan)
 # ArchiTexture: Linguistic-Driven 3D House Plan Generator
 
 ## 🚀 Project Overview
 ArchiTexture is an AI-powered platform that transforms natural language descriptions into detailed 2D and 3D house plans. By leveraging NLP and Machine Learning, it enables architects, designers, and homeowners to generate architectural layouts with minimal effort.
 
 ## ✨ Key Features
 - 🏡 **Text-to-Design Conversion**: Converts natural language descriptions into 2D and 3D layouts.
 - 🔑 **User Authentication**: Secure signup and login functionality.
 - 📐 **Feature Extraction**: Identifies key design elements and spatial relationships.
 - 🖼️ **2D Floor Plan Generation**: Automatically generates structured 2D layouts.
 - 🎨 **3D Model Rendering**: Converts 2D layouts into fully interactive 3D models.
 - 📊 **Constraint Validation**: Ensures logical and functional room arrangements.
 - 💬 **Feedback System**: Users can refine designs based on their preferences.
 
 ## 🛠️ Technologies Used
 ### Frontend
 - **React.js** – Interactive and responsive UI.
 - **Tailwind CSS** – Modern styling framework.
 
 ### Backend
 - **Django** – Powerful web framework for backend logic.
 - **Django REST Framework (DRF)** – API handling.
 - **MySQLL** – Scalable and reliable database management.
 
 ### AI & Machine Learning
 - **Python** – Core language for NLP and ML models.
 - **SpaCy & NLTK** – Natural Language Processing.
 
 ## 🏗️ Installation Guide
 ### Prerequisites
 - Python 3.9+
 - Node.js & npm
 - MySql
 
 ### 🚀 Steps to Set Up
 1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-repo/architexture.git
    cd architexture
    ```
 2. **Backend Setup (Django)**:
    ```sh
    cd backend
    python -m venv env
    source env/bin/activate  # On Windows use: env\Scripts\activate
    pip install -r requirements.txt
    python manage.py migrate
    python manage.py runserver
    ```
 3. **Frontend Setup (React.js)**:
    ```sh
    cd frontend
    npm install
    npm run dev
    ```
 4. **Ensure Database is Running (PostgreSQL)**:
    ```sh
    sudo service postgresql start
    ```
 
 ## 🎯 Usage Guide
 1. Register/Login to access the platform.
 2. Enter a textual description of your desired home layout.
 3. Generate and visualize 2D/3D house plans in real-time.
 4. Provide feedback to refine the generated layouts.
 
 ## 👥 Contributors
 - **Ahmad Hassan** – Lead Developer
 - **Ismail Daniyal** – AI & ML Engineer
 - **Muhammad Zuhair** – Backend Developer (Django)
 
 ## 📜 License
 This project is licensed under the **MIT License**.
 
 💡 *Transforming Ideas into Architecture – One Prompt at a Time!*
