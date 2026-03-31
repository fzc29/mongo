# auth.py
import os
import bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_users_collection():
    client = MongoClient(os.getenv("MONGO_URI_ADMIN"))
    return client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["users"]


def verify_login(username: str, password: str) -> dict | None:
    """
    Returns the user document if credentials are valid, None otherwise.
    Never returns the password hash.
    """
    users = get_users_collection()
    user = users.find_one({"username": username})

    if not user:
        return None

    if bcrypt.checkpw(password.encode(), user["password_hash"]):
        return {
            "username": user["username"],
            "role":     user["role"],
        }

    return None


def is_admin(session_state) -> bool:
    return session_state.get("role") == "admin"


def is_authenticated(session_state) -> bool:
    return session_state.get("authenticated", False)