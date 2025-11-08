from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import datetime, json
from functools import wraps
import jwt
import os
from dotenv import load_dotenv
import pandas as pd
from flask import send_file
import io
# -----------------------------------------
# 🔧 CONFIG
# -----------------------------------------
load_dotenv()
app = Flask(__name__)
# ✅ Use this exact configuration
# CORS(app, 
#      origins=["http://localhost:5173"], 
#      supports_credentials=True,
#      allow_headers=["Content-Type", "Authorization"],
#      methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
client = MongoClient(MONGO_URI)
db = client["Cluster0"]
users_collection = db["users"]
admins_collection = db["admins"]





# -----------------------------------------
# 🔐 JWT Helper Functions
# -----------------------------------------
def create_jwt(email):
    """Generate JWT token with email"""
    token = jwt.encode(
        {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)},
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )
    return token


def token_required(f):
    """Middleware to protect routes using JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
        if not token:
            return jsonify({"error": "Token missing"}), 401

        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user_email = data["email"]
        except Exception as e:
            return jsonify({"error": "Token invalid or expired"}), 401

        return f(current_user_email, *args, **kwargs)
    return decorated

# -----------------------------------------
# 🧩 REGISTER ENDPOINT
# -----------------------------------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    fullName = data.get("fullName")
    email = data.get("email").lower()
    password = data.get("password")
    confirmPassword = data.get("confirmPassword")


    # ✅ Dynamic hybrid sections
    cohortinformation = data.get("cohortinformation", {})
    demographics = data.get("demographics", {})

    

    # ✅ Get users document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        users_doc = {"_id": "users", "batches": []}

    # ✅ Check duplicate email
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            if user["email"] == email:
                return jsonify({"error": "Email already registered"}), 400

    # ✅ Determine which batch to use (every 30 users)
    if not users_doc["batches"]:
        batch_name = "users_batch_1"
        users_doc["batches"].append({"batch_name": batch_name, "users": []})
    else:
        last_batch = users_doc["batches"][-1]
        if len(last_batch["users"]) >= 30:
            batch_name = f"users_batch_{len(users_doc['batches']) + 1}"
            users_doc["batches"].append({"batch_name": batch_name, "users": []})
        else:
            batch_name = last_batch["batch_name"]

    # ✅ Add new user to selected batch
    new_user = {
        "fullName": fullName,
        "email": email,
        "password": password,
        "confirmPassword": confirmPassword,
        "created_at": datetime.datetime.utcnow(),
        "cohortinformation": cohortinformation,
        "demographics": demographics,
        "survey": {},
        "dreamteam": {},
        "bigfive": {}
    }

    for batch in users_doc["batches"]:
        if batch["batch_name"] == batch_name:
            batch["users"].append(new_user)

    # ✅ Save back to MongoDB
    users_collection.replace_one({"_id": "users"}, users_doc, upsert=True)

    token = create_jwt(email)
    return jsonify({
        "token": token,
        "batch": batch_name,
        "user": new_user
    }), 201

# -----------------------------------------
# 🧠 UPDATE PROFILE ENDPOINT (Stepwise updates)
# -----------------------------------------
@app.route("/api/update-profile", methods=["POST"])
@token_required
def update_profile(current_user_email):
    """Update parts of the user (survey, dreamteam, bigfive, etc.)"""
    data = request.json

    # Identify which section to update (dynamic hybrid)
    update_data = {}
    allowed_sections = ["survey", "dreamteam", "bigfive", "cohortinformation", "demographics"]

    for section in allowed_sections:
        if section in data:
            update_data[f"batches.$[batch].users.$[user].{section}"] = data[section]

    if not update_data:
        return jsonify({"error": "No valid section to update"}), 400

    # ✅ Perform the update in nested structure
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": update_data},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"message": "Profile updated successfully"}), 200

# -----------------------------------------
# 🔑 SIGNIN ENDPOINT
# -----------------------------------------
@app.route("/api/signin", methods=["POST"])
def signin():
    """User login using email and password"""
    data = request.get_json()
    email = data.get("email", "").lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # ✅ Find the user in any batch
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or "batches" not in users_doc:
        return jsonify({"error": "No users found"}), 404

    found_user = None
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            if user["email"] == email and user["password"] == password:
                found_user = user
                break
        if found_user:
            break

    if not found_user:
        return jsonify({"error": "Invalid email or password"}), 401

    # ✅ Generate JWT Token
    token = create_jwt(email)

    return jsonify({
        "message": "Signin successful",
        "token": token,
        "user": {
            "fullName": found_user.get("fullName"),
            "email": found_user.get("email"),
            "created_at": found_user.get("created_at")
        }
    }), 200


# -----------------------------------------
# 👤 USER PROFILE ENDPOINT
# -----------------------------------------
@app.route("/api/user-profile", methods=["GET"])
@token_required
def user_profile(current_user_email):
    """
    Fetch the user profile based on the JWT token.
    """
    # Get the users document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return jsonify({"error": "No users found"}), 404

    # Find the user in batches
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            if user["email"] == current_user_email:
                # Return user data except password for security
                user_data = user.copy()
                user_data.pop("password", None)
                user_data.pop("confirmPassword", None)
                return jsonify({"user": user_data, "batch": batch["batch_name"]}), 200

    return jsonify({"error": "User not found"}), 404


# -----------------------------------------
# ADMIN
# -----------------------------------------



# #fetch all users (for testing)
# @app.route("/api/users", methods=["GET"])
# def get_all_users():
#     try:
#         users = list(users_collection.find({}))  # fetch all users

#         # Remove sensitive fields like password
#         for user in users:
#             user["id"] = str(user["_id"])
#             del user["_id"]
#             if "password" in user:
#                 del user["password"]

#         return jsonify({"users": users}), 200

#     except Exception as e:
#         return jsonify({"error": f"Failed to fetch users: {str(e)}"}), 500


# -----------------------------------------
# 👥 FETCH ALL USERS
# -----------------------------------------
@app.route("/api/users", methods=["GET"])
def get_all_users():
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"users": []}), 200

    all_users = []
    for batch in users_doc["batches"]:
        all_users.extend(batch.get("users", []))

    return jsonify({"users": all_users}), 200



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

# -----------------------------
# 🔑 ADMIN SIGNIN
# -----------------------------
@app.route("/api/admin/signin", methods=["POST"])
def admin_signin():
    data = request.json
    email = data.get("email", "").lower()
    password = data.get("password", "")

    admins_collection = db["admins"]
    admin = admins_collection.find_one({"email": email, "password": password})

    if not admin:
        return jsonify({"error": "Invalid email or password"}), 401

    # Create token for admin (optional: can use separate secret if needed)
    token = create_jwt(email)
    return jsonify({"token": token, "admin": {"email": admin["email"], "name": admin.get("name", "")}}), 200

#-----------------------------
# 📋 FETCH ALL USERS FOR ADMIN
# -----------------------------

@app.route("/api/admin/users", methods=["GET"])
def admin_get_all_users():
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return jsonify({"users": []})

    # Flatten all users from batches
    all_users = []
    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            user_copy = user.copy()
            user_copy["batch_name"] = batch.get("batch_name")
            all_users.append(user_copy)

    return jsonify({"users": all_users}), 200


# # Admin Sign-In
# @app.route("/api/admin/signin", methods=["POST"])
# def admin_signin():
#     try:
#         data = request.json
#         email = data.get("email").lower()
#         password = data.get("password")

#         if not email or not password:
#             return jsonify({"success": False, "message": "Email and password required"}), 400

#         # Find admin
#         admin = admins_collection.find_one({"email": email})
#         if not admin or admin.get("password") != password:
#             return jsonify({"success": False, "message": "Invalid credentials"}), 401

#         # Create JWT
#         token = create_jwt(admin["_id"])

#         return jsonify({
#             "success": True,
#             "message": "Login successful",
#             "token": token,
#             "admin": {
#                 "id": str(admin["_id"]),
#                 "firstName": admin.get("firstName"),
#                 "lastName": admin.get("lastName"),
#                 "email": admin.get("email"),
#                 "role": admin.get("role", "admin")
#             }
#         }), 200

#     except Exception as e:
#         return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500




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

# -----------------------------
# 🗑️ ADMIN DELETE SPECIFIC USER
# -----------------------------
@app.route("/api/admin/delete-user", methods=["POST"])
def admin_delete_user():
    data = request.json
    user_email = data.get("email", "").lower()  # email of the user to delete

    if not user_email:
        return jsonify({"error": "User email is required"}), 400

    # Get users document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"error": "No users found"}), 404

    user_deleted = False

    # Iterate over batches to find and delete user
    for batch in users_doc["batches"]:
        for i, user in enumerate(batch["users"]):
            if user["email"] == user_email:
                batch["users"].pop(i)
                user_deleted = True
                break
        if user_deleted:
            break

    if not user_deleted:
        return jsonify({"error": "User not found"}), 404

    # Save updated users back to MongoDB
    users_collection.replace_one({"_id": "users"}, users_doc, upsert=True)

    return jsonify({"message": f"User {user_email} deleted successfully"}), 200


# -----------------------------------------
# 📊 EXPORT ALL USERS TO EXCEL (Admin)
# -----------------------------------------

@app.route("/api/admin/export-users", methods=["GET"])
def export_users():
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"error": "No users found"}), 404

    # Flatten users data
    all_users = []
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            flat_user = {
                "fullName": user.get("fullName"),
                "email": user.get("email"),
            }

            # Cohort information
            cohort = user.get("cohortinformation", {})
            flat_user["programName"] = cohort.get("programName")
            flat_user["programDates"] = cohort.get("programDates")
            flat_user["programVenue"] = cohort.get("programVenue")

            # Demographics
            demo = user.get("demographics", {})
            for key, value in demo.items():
                flat_user[f"demographics_{key}"] = value

            # Survey
            survey = user.get("survey", {})
            for key, value in survey.items():
                flat_user[f"survey_{key}"] = value

            # Dreamteam
            dreamteam = user.get("dreamteam", {})
            for key, value in dreamteam.items():
                flat_user[f"dreamteam_{key}"] = value

            # BigFive
            bigfive = user.get("bigfive", {})
            for key, value in bigfive.items():
                flat_user[f"bigfive_{key}"] = value

            all_users.append(flat_user)

    # Convert to DataFrame
    df = pd.DataFrame(all_users)

    # Save to Excel in memory
    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    # Send as file
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="all_users.xlsx"
    )



# -----------------------------------------
# 🚀 RUN SERVER
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
