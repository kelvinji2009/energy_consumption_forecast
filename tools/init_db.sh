#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Run Database Migrations ---
echo "Running database migrations..."
alembic upgrade head
echo "Database migration complete."

# --- 2. Check if Initialization is Needed ---
echo "Checking if initial data setup is required..."
KEY_COUNT=$(psql "${DATABASE_URL}" -t -A -c "SELECT COUNT(*) FROM api_keys;")

# Check if the command succeeded and the count is 0.
if [ "$?" -eq 0 ] && [ "$KEY_COUNT" -eq 0 ]; then
  echo "No API keys found. Proceeding with first-time initial data setup..."

  # --- 3. Seed Initial Asset Data ---
  echo "--- SEEDING INITIAL ASSET DATA ---"
  python -m database.insert_initial_data
  echo "Initial asset data seeded."

  # --- 4. Generate the Initial API Key ---
  echo "--- GENERATING INITIAL API KEY ---"
  python -m tools.create_initial_key
  echo "--- INITIAL KEY GENERATION COMPLETE ---"

else
  echo "Found ${KEY_COUNT} existing API key(s). Skipping key generation."
  echo "INFO: For security reasons, existing keys cannot be displayed again."
  
  # Get metadata of the most recently created key
  LATEST_KEY_INFO=$(psql "${DATABASE_URL}" -t -A -c "SELECT 'Description: ' || description, 'Created At: ' || created_at FROM api_keys ORDER BY created_at DESC LIMIT 1;")
  
  echo "--- Metadata of most recent key ---"
  echo "${LATEST_KEY_INFO}"
  echo "-----------------------------------"
  echo "If you have lost this key, you must reset the database by running 'docker compose down -v' before starting the application again."
fi

echo "--- INITIALIZATION SCRIPT FINISHED ---"
