# 4cast Server --- FastAPI Backend

A lightweight, production-ready FastAPI backend that provides REST APIs
for managing bets, outcomes, settlements, and blockchain confirmation
logic. Ideal for blockchain-based prediction markets or automated
settlement systems.

------------------------------------------------------------------------

## 🚀 Features

-   ⚡ FastAPI server with auto-generated documentation\
-   🗄️ SQLAlchemy ORM for clean DB modeling\
-   🔍 Pydantic request/response validation\
-   🤖 Autopilot worker for automated polling & settlement\
-   🔐 Environment-based configuration\
-   📘 Interactive API docs at `/docs` (Swagger) and `/redoc`

------------------------------------------------------------------------

## 📦 Project Structure

    app.py                # Main FastAPI app
    db_core.py            # Database initialization
    models.py             # ORM models
    settlement.py         # Settlement logic
    chain_utils.py        # Blockchain utilities
    poll_autopilot.py     # Background worker
    .env.example          # Example environment variables
    requirements.txt      # Dependencies
    README.md             # Documentation

------------------------------------------------------------------------

## 🔧 Installation & Setup

### **1. Clone the Repository**

``` bash
git clone https://github.com/yourusername/4cast-server.git
cd 4cast-server
```

### **2. Create a Virtual Environment**

``` bash
python -m venv venv
source venv/bin/activate     # Linux / macOS
# .\venv\Scripts\activate # Windows
```

### **3. Install Dependencies**

``` bash
pip install -r requirements.txt
```

### **4. Configure Environment Variables**

Create a `.env` file in the project root or copy from example:

``` bash
cp .env.example .env
```

Then update the required values:

    DATABASE_URL=sqlite:///./aster.db
    MIN_BET_BNB=0.01
    CONFIRMATIONS=1

------------------------------------------------------------------------

## ▶️ Run the Server Locally

``` bash
uvicorn app:app --reload
```

Server will start at:

-   API Base → http://127.0.0.1:8000\
-   Swagger UI → http://127.0.0.1:8000/docs\
-   ReDoc → http://127.0.0.1:8000/redoc


------------------------------------------------------------------------

# 🧰 Tech Stack

-   **Python 3.9+**\
-   **FastAPI**\
-   **SQLAlchemy**\
-   **Pydantic**\
-   **Uvicorn**\
-   **dotenv**\
-   Optional: **web3.py** (depending on blockchain usage)

------------------------------------------------------------------------

# 🤝 Contributing

Contributions are welcome!

### **Steps:**

1.  Fork the repository\

2.  Create a new branch:

    ``` bash
    git checkout -b feature-name
    ```

3.  Make your changes\

4.  Run tests & linting\

5.  Submit a pull request

### **Guidelines**

-   Follow clean code principles\
-   Ensure PRs are descriptive\
-   Keep commits atomic\
-   Add documentation if introducing new APIs

------------------------------------------------------------------------

# 📄 License

This project is licensed under the **MIT License**.\
You are free to use, modify, and distribute it with attribution.

