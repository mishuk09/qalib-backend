from flask import Flask, request, jsonify,send_file,send_from_directory
from flask_cors import CORS, cross_origin
import datetime, json
from functools import wraps
import jwt
import os
from dotenv import load_dotenv
import pandas as pd
from flask import send_file
import io
import numpy as np
# from pso import run_pso, load_dataset, comput e_group_score
from pso import run_pso, load_dataset, compute_group_score_from_values
from werkzeug.utils import secure_filename
import time
from routes.post import post_bp
from config.db import db
from utils.jwt_auth import token_required


# -----------------------------------------
# 🔧 CONFIG
# -----------------------------------------
load_dotenv()
app = Flask(__name__)
 

app.register_blueprint(post_bp, url_prefix="/api")

# ✅ Enable full CORS for all API routes
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)


# ✅ Handle preflight OPTIONS requests (important for axios + file upload)
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return '', 200


app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")


def delete_file_if_exists(file_path):
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print("Failed to delete file:", e)


# MongoDB connection
JWT_SECRET = os.getenv("JWT_SECRET")

users_collection = db["users"]
admins_collection = db["admins"]

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

UPLOAD_ROOT = "uploads"
PROFILE_FOLDER = os.path.join(UPLOAD_ROOT, "profile")
COVER_FOLDER = os.path.join(UPLOAD_ROOT, "cover")

os.makedirs(PROFILE_FOLDER, exist_ok=True)
os.makedirs(COVER_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_MB = 5

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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

 

# -----------------------------------------
# PSO 
# -----------------------------------------

@app.route("/api/admin/pso-run", methods=["POST", "OPTIONS"])
@cross_origin()
def run_pso_api():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, "dataset.xlsx")
        file.save(file_path)

        # Read original excel and prepare the same reduced df_new you had before
        df = pd.read_excel(file_path)
        # Keep same slicing as original
        df_subset = df.iloc[:, 39:140]

        D = df_subset.iloc[:, 0:25].sum(axis=1)
        H = df_subset.iloc[:, 25:48].sum(axis=1)
        T = df_subset.iloc[:, 48:63].sum(axis=1)
        DT1 = df_subset.iloc[:, 77:82].sum(axis=1)
        DT2 = df_subset.iloc[:, 83:88].sum(axis=1)
        DT3 = df_subset.iloc[:, 89:94].sum(axis=1)

        df_new = pd.DataFrame(
            {"D": D, "H": H, "T": T, "DT1": DT1, "DT2": DT2, "DT3": DT3}
        )

        # Save reduced dataset (like original flow)
        dataset_path = os.path.join(UPLOAD_FOLDER, "dataset.xlsx")
        df_new.to_excel(dataset_path, index=False)

        # Load dataset once via pso.load_dataset (this ensures numeric-only and float32 conversion)
        df_loaded = load_dataset(dataset_path)  # uses optimized loader

        # Run PSO with loaded df
        pos, val, hist, groups = run_pso(
            df_loaded, max_iter=100, num_particles=30, n_mem=3, scoring="min", verbose=False
        )

        # Compute per-group fit quickly using numpy
        arr_vals = df_loaded.values
        fit = [compute_group_score_from_values(arr_vals, members) for members in groups.values()]

        # Internal best index (0-based, for groups dict)
        if len(fit) > 0:
            best_group_pos = int(np.argmax(fit))   # 0-based internal index
            best_score = float(np.max(fit))
        else:
            best_group_pos = 0
            best_score = float(val)

        # Display group number should start from 1
        best_group_number = best_group_pos + 1

        # -----------------------------
        # Build output DataFrame with names
        # Group number in file: 1, 2, 3, ...
        # Score column is commented out / removed
        # -----------------------------
        data = []
        for display_idx, (g, members) in enumerate(groups.items(), start=1):
            members_names = ", ".join(df.iloc[members]["fullName"].astype(str))
            # If you want to keep score in memory but not in file:
            # score_value = round(fit[display_idx - 1], 2)
            # data.append((display_idx, score_value, members_names))
            data.append((display_idx, members_names))

        # Only Group and Members in df_grp (no Score column)
        df_grp = pd.DataFrame(data, columns=["Group", "Members"])

        # Save as TXT (tab-separated) with only Group + Members
        output_path = os.path.join(OUTPUT_FOLDER, "group_final.txt")
        df_grp.to_csv(output_path, index=False, sep="\t")

        return jsonify({
            "status": "success",
            # Return display group number starting from 1
            "best_group_index": int(best_group_number),
            "best_group": groups[best_group_pos],
            "best_score": float(best_score),
            "download_url": "/api/admin/download/group_final.txt",
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500



@app.route("/api/admin/download/<path:filename>", methods=["GET", "OPTIONS"])
@cross_origin()
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

# Updated Backend Endpoints for Profile and Cover Photo Upload
# Replace your old endpoints with these updated versions


@app.route("/api/user/upload-profile-photo", methods=["POST"])
@cross_origin()
@token_required
def upload_profile_photo(current_user_email):
    
    data = request.get_json()
    
    if not data or "profilePhotoUrl" not in data:
        return jsonify({"error": "No photo URL provided"}), 400
    
    profile_photo_url = data["profilePhotoUrl"]
    
    # Normalize email for matching
    current_user_email = (current_user_email or "").lower()
    print(f"🔍 Uploading profile photo for: {current_user_email}")

    # ✅ Use MongoDB array filters (same as add-survey)
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": {
            "batches.$[batch].users.$[user].profilePhoto": {
                "url": profile_photo_url,
                "path": profile_photo_url
            }
        }},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )
    
    print(f"💾 Database update - Modified: {result.modified_count}")
    
    if result.modified_count == 0:
        print(f"❌ User not found for: {current_user_email}")
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "message": "Profile photo uploaded successfully",
        "profilePhotoUrl": profile_photo_url,
        "userEmail": current_user_email
    }), 200


@app.route("/api/user/upload-cover-photo", methods=["POST"])
@cross_origin()
@token_required
def upload_cover_photo(current_user_email):
    """
    Upload cover photo to Cloudinary and store URL in database
    """
    data = request.get_json()
    
    if not data or "coverPhotoUrl" not in data:
        return jsonify({"error": "No photo URL provided"}), 400
    
    cover_photo_url = data["coverPhotoUrl"]
    
    # Normalize email for matching
    current_user_email = (current_user_email or "").lower()
    print(f"🔍 Uploading cover photo for: {current_user_email}")

    # ✅ Use MongoDB array filters (same as add-survey)
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": {
            "batches.$[batch].users.$[user].coverPhoto": {
                "url": cover_photo_url,
                "path": cover_photo_url
            }
        }},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )
    
    print(f"💾 Database update - Modified: {result.modified_count}")
    
    if result.modified_count == 0:
        print(f"❌ User not found for: {current_user_email}")
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "message": "Cover photo uploaded successfully",
        "coverPhotoUrl": cover_photo_url,
        "userEmail": current_user_email
    }), 200


# user search functionality

@app.route("/api/search-users", methods=["GET"])
def search_users():
    query = request.args.get("query", "").strip()

    if not query:
        return jsonify({"users": []}), 200

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"users": []}), 200

    query_lower = query.lower()
    matched_users = []

    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            full_name = user.get("fullName", "").lower()

            if query_lower in full_name:
                user_copy = user.copy()

                # remove sensitive fields
                user_copy.pop("password", None)
                user_copy.pop("confirmPassword", None)

                user_copy["batch_name"] = batch.get("batch_name")
                matched_users.append(user_copy)

    return jsonify({
        "query": query,
        "count": len(matched_users),
        "users": matched_users
    }), 200



# -----------------------------------------
# 🧠 UPDATE survey
# -----------------------------------------
@app.route("/api/update-survey", methods=["POST"])
@token_required
def update_profile(current_user_email):
    """
    Update parts of the user (survey, dreamteam, bigfive, etc.)
    """
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
# 👤 UPDATE PROFILE ENDPOINT
# -----------------------------------------
# @app.route("/api/update-profile", methods=["PUT"])
# @token_required
# def update_profile(current_user_email):
#     """
#     Update user profile: fullName, email, cohortinformation, demographics.
#     """
#     data = request.get_json()
    
#     # Extract the four specific fields
#     new_full_name = data.get("fullName")
#     new_email = data.get("email")
#     new_cohort = data.get("cohortinformation")
#     new_demo = data.get("demographics")

#     # 1. Fetch the main users document
#     users_doc = users_collection.find_one({"_id": "users"})
#     if not users_doc:
#         return jsonify({"error": "No users document found"}), 404

#     user_found = False
    
#     # 2. Iterate through batches to find the specific user
#     for batch in users_doc.get("batches", []):
#         for user in batch.get("users", []):
#             if user.get("email") == current_user_email:
                
#                 # Check if the new email is already taken by someone else
#                 if new_email and new_email.lower() != current_user_email:
#                     # Logic to check for email uniqueness across all batches
#                     for b in users_doc["batches"]:
#                         if any(u["email"] == new_email.lower() for u in b["users"]):
#                             return jsonify({"error": "The new email is already registered"}), 400
                
#                 # 3. Apply updates only if data is provided
#                 if new_full_name:
#                     user["fullName"] = new_full_name
#                 if new_email:
#                     user["email"] = new_email.lower()
#                 if new_cohort is not None:
#                     # Merge cohortinformation (preserve existing keys if not provided)
#                     if not isinstance(user.get("cohortinformation"), dict):
#                         user["cohortinformation"] = {}
#                     user["cohortinformation"].update(new_cohort)
#                 if new_demo is not None:
#                     # Merge demographics (preserve existing keys if not provided)
#                     if not isinstance(user.get("demographics"), dict):
#                         user["demographics"] = {}
#                     user["demographics"].update(new_demo)
                
#                 user_found = True
#                 break
#         if user_found:
#             break

#     if not user_found:
#         return jsonify({"error": "User not found"}), 404

#     # 4. Save the entire updated document back to MongoDB
#     try:
#         result = users_collection.replace_one({"_id": "users"}, users_doc, upsert=True)
#         if result.matched_count == 0 and result.upserted_id is None:
#             return jsonify({"error": "Failed to update user profile in database"}), 500
#     except Exception as e:
#         print("Database update error:", e)
#         return jsonify({"error": f"Database error: {str(e)}"}), 500

#     # 5. If the email changed, we need to issue a new JWT token
#     new_token = None
#     if new_email and new_email.lower() != current_user_email:
#         new_token = create_jwt(new_email.lower())

#     return jsonify({
#         "message": "Profile updated successfully",
#         "token": new_token if new_token else None,
#         "updatedFields": {
#             "fullName": new_full_name,
#             "email": new_email,
#             "cohortinformation": new_cohort,
#             "demographics": new_demo
#         }
#     }), 200



# -----------------------------------------
# 🧠 ADD SURVEY
# -----------------------------------------
@app.route("/api/add-survey", methods=["POST"])
@token_required
def add_survey(current_user_email):
    """
    Update parts of the user (add survey, dreamteam, bigfive, etc.)
    """
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
# 👤 USER SURVEY ENDPOINT
# -----------------------------------------
@app.route("/api/user-survey", methods=["GET"])
@token_required
def user_survey(current_user_email):
    """
    Fetch the logged-in user's survey data (D1, D2..., H1, H2...) based on JWT token.
    """
    # Get the users document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return jsonify({"error": "No users found"}), 404

    # Find the user in batches
    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            if user.get("email") == current_user_email:
                survey_data = user.get("survey", {})  # only survey data
                if not survey_data:
                    return jsonify({"error": "Survey data not found"}), 404

                return jsonify({
                    "survey": survey_data,
                    "batch": batch.get("batch_name")
                }), 200

    return jsonify({"error": "User not found"}), 404


# -----------------------------------------
# ADMIN
# -----------------------------------------


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




@app.route("/api/admin/users-by-program", methods=["GET"])
def admin_users_by_program():
    """
    Return a list of all programs and users grouped by programName.
    Response:
      {
        "programs": ["All", "Program A", "Program B", ...],
        "usersByProgram": {
           "All": [...],
           "Program A": [...],
           ...
        }
      }
    """
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"programs": ["All"], "usersByProgram": {"All": []}}), 200

    all_users = []
    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            user_copy = user.copy()
            user_copy["batch_name"] = batch.get("batch_name")
            all_users.append(user_copy)

    # Group by programName (found under cohortinformation.programName)
    users_by_program = {}
    users_by_program["All"] = all_users

    for user in all_users:
        cohort = user.get("cohortinformation", {}) or {}
        program = cohort.get("programName") or "Unknown Program"
        if program not in users_by_program:
            users_by_program[program] = []
        users_by_program[program].append(user)

    # Build programs list in stable order: All first, then sorted program names
    extra_programs = [p for p in users_by_program.keys() if p != "All"]
    # keep ordering consistent (Unknown Program at end)
    sorted_programs = sorted([p for p in extra_programs if p != "Unknown Program"]) 
    if "Unknown Program" in extra_programs:
        sorted_programs.append("Unknown Program")
    programs_list = ["All"] + sorted_programs

    return jsonify({"programs": programs_list, "usersByProgram": users_by_program}), 200


# Update existing export endpoint to accept optional programName query param:
@app.route("/api/admin/export-users", methods=["GET"])
def export_users():
    # optional programName query param
    program_filter = request.args.get("programName")  # can be None or "All" or a specific program

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"error": "No users found"}), 404

    # Flatten users data and optionally filter by programName
    all_users = []
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            cohort = user.get("cohortinformation", {}) or {}
            program_name = cohort.get("programName") or "Unknown Program"

            # Apply filter if provided and not "All"
            if program_filter and program_filter != "All" and program_filter != program_name:
                continue

            flat_user = {
                "fullName": user.get("fullName"),
                "email": user.get("email"),
                "programName": program_name,
            }

            # Cohort information (kept)
            flat_user["programDates"] = cohort.get("programDates")
            flat_user["programVenue"] = cohort.get("programVenue")

            # Demographics
            demo = user.get("demographics", {}) or {}
            for key, value in demo.items():
                flat_user[f"demographics_{key}"] = value

            # Survey
            survey = user.get("survey", {}) or {}
            for key, value in survey.items():
                flat_user[f"survey_{key}"] = value

            # Dreamteam
            dreamteam = user.get("dreamteam", {}) or {}
            for key, value in dreamteam.items():
                flat_user[f"dreamteam_{key}"] = value

            # BigFive
            bigfive = user.get("bigfive", {}) or {}
            for key, value in bigfive.items():
                flat_user[f"bigfive_{key}"] = value

            all_users.append(flat_user)

    # Convert to DataFrame
    df = pd.DataFrame(all_users)

    # Save to Excel in memory
    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    # Build download_name to indicate program when filtered
    download_name = "all_users.xlsx"
    if program_filter and program_filter != "All":
        # sanitize program_filter for filename (simple replacement)
        safe_name = program_filter.replace(" ", "_").replace("/", "_")
        download_name = f"users_{safe_name}.xlsx"

    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=download_name
    )
 

@app.route("/api/media/profile/<filename>", methods=["GET"])
def get_profile_photo(filename):
    return send_from_directory(PROFILE_FOLDER, filename)

@app.route("/api/media/cover/<filename>", methods=["GET"])
def get_cover_photo(filename):
    return send_from_directory(COVER_FOLDER, filename)



# -----------------------------------------
# 🚀 RUN SERVER
# -----------------------------------------

# --- Production vs. Development Startup ---
if __name__ == '__main__':
    # This block is only for local development testing. 
    # Render's Gunicorn command will ignore this block.
    # We ensure it binds correctly if run locally, though 
    # typically you don't run it this way in production.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)