FROM python:3.10

# Set up user for better security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories for PDF storage
RUN mkdir -p /app/app
USER root
RUN chown -R user:user /app
USER user

# Copy the application code
COPY --chown=user app /app/app

# Set default environment variable for project name
ENV OPIK_PROJECT_NAME="cuny-suny-assistant-chatbot-chat"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
