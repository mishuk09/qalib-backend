from flask import Flask, request, jsonify
from flask_cors import cross_origin

from pymongo import MongoClient
from bson import ObjectId
import jwt
import datetime
import os
from dotenv import load_dotenv
from bson import ObjectId


# Load env vars
load_dotenv()

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    return response

# Config
MONGO_URI = os.getenv("MONGO_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
client = MongoClient(MONGO_URI)
db = client["Cluster0"]
users_collection = db["users"]
admins_collection = db["admins"]


# Helper: Create JWT
def create_jwt(user_id):
    payload = {
        "user_id": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

# Helper: Verify JWT
def verify_jwt(token):
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# Register
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email").lower()
    password = data.get("password")

    # demographics
    demographics = {
        "religion": data.get("religion"),
        "gender": data.get("gender"),
        "age": data.get("age"),
        "place_of_residence": data.get("place_of_residence"),
        "father_occupation": data.get("father_occupation"),
        "mother_occupation": data.get("mother_occupation"),
        "household_monthly_income": data.get("household_monthly_income"),
        "education_level": data.get("education_level"),
        "field_of_study": data.get("field_of_study"),
        "university_college_name": data.get("university_college_name"),
        "attended_government_program": data.get("attended_government_program"),
        "has_entrepreneur_family_or_friends": data.get("has_entrepreneur_family_or_friends"),
        "currently_entrepreneur": data.get("currently_entrepreneur"),
        "prior_entrepreneurship_experience": data.get("prior_entrepreneurship_experience"),
        "considered_inclusive_entrepreneur": data.get("considered_inclusive_entrepreneur"),
    }

    # behavior data (force dict)
    behavior_data = data.get("behavior_data", {})
    if isinstance(behavior_data, str):
        import json
        try:
            behavior_data = json.loads(behavior_data)
        except:
            behavior_data = {}

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 400

    user_id = users_collection.insert_one({
        "name": name,
        "email": email,
        "password": password,
        "created_at": datetime.datetime.utcnow(),
        "demographics": demographics,
        "behavior_data": behavior_data
    }).inserted_id

    token = create_jwt(user_id)

    return jsonify({
        "token": token,
        "user": {
            "id": str(user_id),
            "name": name,
            "email": email,
            "demographics": demographics,
            "behavior_data": behavior_data
        }
    })


# Login
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email").lower()
    password = data.get("password")

    user = users_collection.find_one({"email": email})
    if not user or user["password"] != password:   # plain text comparison
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_jwt(user["_id"])
    return jsonify({
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"]
        }
    })


#fetch all users (for testing)
@app.route("/api/users", methods=["GET"])
def get_all_users():
    try:
        users = list(users_collection.find({}))  # fetch all users

        # Remove sensitive fields like password
        for user in users:
            user["id"] = str(user["_id"])
            del user["_id"]
            if "password" in user:
                del user["password"]

        return jsonify({"users": users}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch users: {str(e)}"}), 500




# Delete user by ID
@app.route("/api/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    try:
        result = users_collection.delete_one({"_id": ObjectId(user_id)})
        
        if result.deleted_count == 0:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        return jsonify({"success": True, "message": f"User {user_id} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"success": False, "message": f"Error deleting user: {str(e)}"}), 500





# Admin Signup
@app.route("/api/admin/signup", methods=["POST"])
def admin_signup():
    try:
        data = request.json
        first_name = data.get("firstName")
        last_name = data.get("lastName")
        email = data.get("email").lower()
        password = data.get("password")  # ⚠️ stored as plain text

        if not first_name or not last_name or not email or not password:
            return jsonify({"success": False, "message": "All fields are required"}), 400

        # Check if admin already exists
        if admins_collection.find_one({"email": email}):
            return jsonify({"success": False, "message": "Email already registered"}), 400

        # Insert new admin
        admin_id = admins_collection.insert_one({
            "firstName": first_name,
            "lastName": last_name,
            "email": email,
            "password": password,  # ⚠️ plain text
            "role": "admin",
            "created_at": datetime.datetime.utcnow()
        }).inserted_id

        # Create JWT
        token = create_jwt(admin_id)

        return jsonify({
            "success": True,
            "message": "Admin registered successfully",
            "token": token,
            "admin": {
                "id": str(admin_id),
                "firstName": first_name,
                "lastName": last_name,
                "email": email,
                "role": "admin"
            }
        }), 201

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500



# Admin Sign-In
@app.route("/api/admin/signin", methods=["POST"])
def admin_signin():
    try:
        data = request.json
        email = data.get("email").lower()
        password = data.get("password")

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password required"}), 400

        # Find admin
        admin = admins_collection.find_one({"email": email})
        if not admin or admin.get("password") != password:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

        # Create JWT
        token = create_jwt(admin["_id"])

        return jsonify({
            "success": True,
            "message": "Login successful",
            "token": token,
            "admin": {
                "id": str(admin["_id"]),
                "firstName": admin.get("firstName"),
                "lastName": admin.get("lastName"),
                "email": admin.get("email"),
                "role": admin.get("role", "admin")
            }
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500




#fetch all admins (for testing)
@app.route("/api/admin", methods=["GET"])
def get_all_admins():
    try:
        admins = list(admins_collection.find({}))  # fetch all admins

        # Remove sensitive fields like password
        for admin in admins:
            admin["id"] = str(admin["_id"])
            del admin["_id"]
            if "password" in admin:
                del admin["password"]

        return jsonify({"admins": admins}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch admins: {str(e)}"}), 500




# Google Login (placeholder)
@app.route("/api/login/google", methods=["POST"])
def google_login():
    data = request.json
    google_email = data.get("email")
    name = data.get("name")

    user = users_collection.find_one({"email": google_email})
    if not user:
        user_id = users_collection.insert_one({
            "name": name,
            "email": google_email,
            "google_account": True,
            "created_at": datetime.datetime.utcnow()
        }).inserted_id
    else:
        user_id = user["_id"]

    token = create_jwt(user_id)
    return jsonify({"token": token, "user": {"id": str(user_id), "name": name, "email": google_email}})

# X Login (placeholder)
@app.route("/api/login/x", methods=["POST"])
def x_login():
    data = request.json
    x_email = data.get("email")
    name = data.get("name")

    user = users_collection.find_one({"email": x_email})
    if not user:
        user_id = users_collection.insert_one({
            "name": name,
            "email": x_email,
            "x_account": True,
            "created_at": datetime.datetime.utcnow()
        }).inserted_id
    else:
        user_id = user["_id"]

    token = create_jwt(user_id)
    return jsonify({"token": token, "user": {"id": str(user_id), "name": name, "email": x_email}})

# Profile (Protected Route)
@app.route("/api/profile", methods=["GET"])
def profile():
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "No token"}), 401

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return jsonify({"error": "Invalid auth header format"}), 401

    token = parts[1]
    decoded = verify_jwt(token)
    if not decoded:
        return jsonify({"error": "Invalid or expired token"}), 401

    user = users_collection.find_one({"_id": ObjectId(decoded["user_id"])})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Remove sensitive data like password before sending response
    user.pop("password", None)

    return jsonify({"user": {**user, "id": str(user["_id"])}})

# Update Profile
@app.route("/api/update-profile", methods=["POST"])
def update_profile():
    try:
        data = request.get_json()
        email = data.get("email")

        if not email:
            return jsonify({"success": False, "message": "Email is required"}), 400

        # Do not allow password updates here
        update_data = {k: v for k, v in data.items() if k != "password" and v != ""}

        result = users_collection.update_one(
            {"email": email},
            {"$set": update_data},
            upsert=False
        )

        if result.matched_count == 0:
            return jsonify({"success": False, "message": "User not found"}), 404

        return jsonify({
            "success": True,
            "message": f"Profile updated successfully for {email}",
            "updated_fields": update_data
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error updating profile: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
