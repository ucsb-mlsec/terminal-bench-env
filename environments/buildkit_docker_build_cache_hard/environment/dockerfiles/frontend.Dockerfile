FROM node:18

WORKDIR /app

# NOTE: Rebuild happens too often here
# Problem: Copying all source code before installing dependencies
COPY . /app

# This npm install runs every time any source file changes
RUN npm install

# NOTE: No cache mount used for npm cache directory
RUN npm run build

# Expose port for the application
EXPOSE 3000

# Start the application
CMD ["npm", "start"]