#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Run Database Migrations ---
# This should always run to ensure the schema is up-to-date.
echo "Running database migrations..."
alembic upgrade head
echo "Database migration complete."

# --- 2. Check if Initialization is Needed ---
# We check if any API keys already exist. If they do, we skip all initial data setup.
echo "Checking if initial data setup is required..."
# The -t flag is for tuples-only (no headers/footers), -A is for unaligned (no padding).
KEY_COUNT=$(psql "${DATABASE_URL}" -t -A -c "SELECT COUNT(*) FROM api_keys;")

# Check if the command succeeded and the count is 0.
if [ "$?" -eq 0 ] && [ "$KEY_COUNT" -eq 0 ]; then
  echo "No API keys found. Proceeding with first-time initial data setup..."

  # --- 3. Seed Initial Asset Data ---
  # This only runs if the database is completely new.
  echo "--- SEEDING INITIAL ASSET DATA ---"
  python -m database.insert_initial_data
  echo "Initial asset data seeded."

  # --- 4. Generate the Initial API Key ---
  echo "--- GENERATING INITIAL API KEY ---"
  python -m tools.create_initial_key
  echo "--- INITIAL KEY GENERATION COMPLETE ---"

else
  echo "Found ${KEY_COUNT} existing API key(s). Skipping initial data setup."
fi

echo "--- INITIALIZATION SCRIPT FINISHED ---"