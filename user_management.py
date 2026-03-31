# user_management.py
import os
import bcrypt
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def get_users_collection():
    client = MongoClient(
        os.getenv("MONGO_URI"),
        tlsCAFile=certifi.where()
    )
    db = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]
    collection = db["users"]
    # Create index here inside the function, not at module level
    collection.create_index("username", unique=True)
    return collection


def create_user(username: str, password: str, role: str = "guest"):
    users = get_users_collection()
    if users.find_one({"username": username}):
        print(f"User '{username}' already exists.")
        return
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "username": username,
        "password_hash": hashed,
        "role": role,
        "created_at": datetime.utcnow(),
    })
    print(f"Created {role} user: {username}")


def delete_user(username: str):
    users = get_users_collection()
    result = users.delete_one({"username": username})
    if result.deleted_count:
        print(f"Deleted user: {username}")
    else:
        print(f"User '{username}' not found.")


def list_users():
    users = get_users_collection()
    print(f"\n{'Username':<20} {'Role':<10}")
    print("-" * 30)
    for u in users.find({}, {"username": 1, "role": 1}):
        print(f"  {u['username']:<20} {u['role']:<10}")


def change_role(username: str, new_role: str):
    users = get_users_collection()
    result = users.update_one(
        {"username": username},
        {"$set": {"role": new_role}}
    )
    if result.modified_count:
        print(f"Role updated for {username} → {new_role}")
    else:
        print(f"User '{username}' not found.")


def change_password(username: str, new_password: str):
    users = get_users_collection()
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
    result = users.update_one(
        {"username": username},
        {"$set": {"password_hash": hashed}}
    )
    if result.modified_count:
        print(f"Password updated for: {username}")
    else:
        print(f"User '{username}' not found.")
