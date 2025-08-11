#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Wait for Database to be ready ---
echo "Waiting for database to be ready..."
# The 'db-init' service in docker-compose already has 'depends_on: db',
# so we can proceed directly.

# --- 2. Run Database Migrations ---
echo "Running database migrations..."
alembic upgrade head
echo "Database migration complete."

# --- 3. Truncate API Keys Table ---
# This ensures a clean state every time the environment is rebuilt.
echo "--- TRUNCATING API KEYS TABLE ---"
echo "TRUNCATE TABLE api_keys RESTART IDENTITY CASCADE;" | psql "${DATABASE_URL}"
echo "API keys table truncated."

# --- 4. Seed Initial Asset Data ---
echo "--- SEEDING INITIAL ASSET DATA ---"
python -m database.insert_initial_data
echo "Initial asset data seeded."

# --- 5. Generate Initial API Key ---
# The key will be printed to the docker logs for the user to copy.
echo "--- GENERATING INITIAL API KEY ---"
python -m tools.create_initial_key
echo "--- INITIALIZATION COMPLETE ---"
