#!/bin/bash
# Create required directories for PostgreSQL
mkdir -p /var/run/postgresql
chmod 775 /var/run/postgresql

# Set PATH to include PostgreSQL binaries
export PATH=/usr/lib/postgresql/16/bin:$PATH

echo "Starting PostgreSQL on $(hostname)..."
echo "Data directory: /var/lib/postgresql/data"
echo "Listening on: 0.0.0.0:5432"

# Start PostgreSQL (running as current user, not root)
exec postgres -D /var/lib/postgresql/data -h 0.0.0.0 -p 5432
