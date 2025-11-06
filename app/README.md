# BISEE App Backend

A backend application for the BISEE project, built for college coursework using Python and Uvicorn.

## Features

- RESTful API endpoints
- Database integration
- Authentication and authorization
- Error handling and logging

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/bisee-app.git
    cd bisee-app
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Copy `.env.example` to `.env`
    - Fill in your database credentials and API keys

4. Run the application:
    ```bash
    uvicorn app.main:app --reload
    ```

## Usage

The server will start on `http://localhost:8000`. Use tools like Postman to test the API endpoints.

### API Endpoints

- `GET /api/v1/auth/google/url` - Handle sign-in with Google
- `POST /api/v1/auth/google/callback` - Handle getting access token from Google callback
- `POST /api/v1/rag-chat/rag` - Handle chat with request body containing message and session_id

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.
