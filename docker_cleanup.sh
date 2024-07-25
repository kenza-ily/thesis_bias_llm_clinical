#!/bin/bash

# Remove all stopped containers
docker container prune -f

# Remove all dangling images
docker image prune -f

# Remove all unused volumes
docker volume prune -f

# Remove all unused networks
docker network prune -f

# Remove all unused objects
docker system prune -f

# Show disk usage
docker system df