from flask import Flask, request, jsonify,send_file,send_from_directory
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
import datetime, json
from functools import wraps
import jwt
import os
from dotenv import load_dotenv
import pandas as pd
from flask import send_file
import io
import numpy as np
from pso import run_pso, load_dataset, compute_group_score



# -----------------------------------------
# 🔧 CONFIG
# -----------------------------------------
load_dotenv()
app = Flask(__name__)
 
# CORS(app, resources={r"/*": {"origins": "*"}})
# The best practice for production and for supporting credentials
# CORS(app, supports_credentials=True, resources={
#     r"/*": {
#         "origins": [
#             "http://localhost:3000",
#             "http://localhost:5173",
#             "https://qalib.org",
#             "https://orangered-fox-171828.hostingersite.com"
#         ]
#     }
# })

# ✅ Enable full CORS for all API routes
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)


# ✅ Handle preflight OPTIONS requests (important for axios + file upload)
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return '', 200


app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")

# MongoDB connection


# Initialize DB connection (Note: Always ensure MONGO_URI is set on Render!)
# MONGO_URI = os.getenv("MONGO_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
# client = MongoClient(MONGO_URI)
# db = client["Cluster0"]

# if MONGO_URI:
#     client = MongoClient(MONGO_URI)
#     db = client.get_database('Cluster0')
# else:
#     print("WARNING: MONGO_URI not set. Database operations will fail.")
#     db = None



 
MONGO_URI = os.environ.get('MONGO_URI')

# Initialize DB connection (DB name explicitly set to 'Cluster0')
if MONGO_URI:
    try:
        # Attempt connection using the URI from environment variables
        client = MongoClient(MONGO_URI)
        # Setting the database name as 'Cluster0'
        db = client.get_database('Cluster0')
        print("MongoDB connection established to database 'Cluster0'.")
    except Exception as e:
        print(f"ERROR: Failed to connect to MongoDB: {e}")
        db = None
else:
    print("WARNING: MONGO_URI environment variable is not set. Database operations will fail.")
    db = None






users_collection = db["users"]
admins_collection = db["admins"]





UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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

        df = pd.read_excel(file_path)
        num_rows = df.shape[0]
        df_subset = df.iloc[:, 39:140]

        D = df_subset.iloc[:, 0:25].sum(axis=1)
        H = df_subset.iloc[:, 25:48].sum(axis=1)
        T = df_subset.iloc[:, 48:63].sum(axis=1)
        DT1 = df_subset.iloc[:, 77:82].sum(axis=1)
        DT2 = df_subset.iloc[:, 83:88].sum(axis=1)
        DT3 = df_subset.iloc[:, 89:94].sum(axis=1)

        df_new = pd.DataFrame({'D': D, 'H': H, 'T': T, 'DT1': DT1, 'DT2': DT2, 'DT3': DT3})
        dataset_path = os.path.join(UPLOAD_FOLDER, "dataset.xlsx")
        df_new.to_excel(dataset_path, index=False)

        pos, val, hist, groups = run_pso(max_iter=100, num_particles=30)
        df_loaded = load_dataset()
        fit = [compute_group_score(df_loaded, members) for members in groups.values()]
        best_score = np.max(fit)
        best_group_index = np.argmax(fit)

        data = zip(groups.items(), fit)
        df_grp = pd.DataFrame(data, columns=["Group", "Score"])
        output_path = os.path.join(OUTPUT_FOLDER, "group_final.xlsx")
        df_grp.to_excel(output_path, index=False)

        return jsonify({
            "status": "success",
            "best_group_index": int(best_group_index),
            "best_group": groups[best_group_index],
            "best_score": float(best_score),
            "download_url": "/api/admin/download/group_final.xlsx"
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


# @app.route("/api/admin/download/<path:filename>")
# def download_file(filename):
#     """Serve the generated Excel file for download."""
#     return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route("/api/admin/download/<path:filename>", methods=["GET", "OPTIONS"])
@cross_origin()
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)



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