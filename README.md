# Real-Time Sentiment Analysis Pipeline

A production-ready FastAPI-based sentiment analysis pipeline with real-time predictions, caching, and database integration.

## Project Structure

```
sentiment-pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── database.py          # PostgreSQL connection
│   ├── cache.py             # Redis connection
│   └── ml/
│       ├── __init__.py
│       ├── train.py         # Model training script
│       ├── features.py      # Feature engineering
│       └── predict.py       # Prediction logic
├── data/
│   └── IMDB Dataset.csv     # Download from Kaggle
├── models/
│   └── sentiment_model.pkl  # Trained model
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Features

- **Real-time Sentiment Analysis**: Fast predictions using pre-trained models
- **REST API**: FastAPI with full CORS support
- **Caching Layer**: Redis for improved performance
- **Database Integration**: PostgreSQL for storing predictions and history
- **Docker Support**: Complete Docker and Docker Compose setup
- **Feature Engineering**: TF-IDF vectorization and statistical features
- **Batch Processing**: Support for multiple predictions at once

## Prerequisites

- Docker and Docker Compose (for containerized setup)
- Python 3.11+ (for local development)
- PostgreSQL 15+ (if running locally)
- Redis 7+ (if running locally)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-pipeline

# Build and start all services
docker-compose up --build

# The API will be available at http://localhost:8000
# API documentation: http://localhost:8000/docs
```

### Local Development

1. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
```

4. **Start PostgreSQL and Redis** (using Docker):
```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15-alpine
docker run -d -p 6379:6379 redis:7-alpine
```

5. **Run the application**:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Sentiment Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "This movie was absolutely amazing!"
}
```

### Batch Sentiment Prediction
```bash
POST /predict-batch
Content-Type: application/json

{
  "texts": [
    "Great product!",
    "Terrible experience.",
    "It's okay."
  ]
}
```

## Training the Model

To train the model on the IMDB dataset:

```bash
python -m app.ml.train
```

Make sure the dataset is in the `data/` directory.

## Configuration

Environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_HOST`: Redis hostname (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `LOG_LEVEL`: Logging level (default: INFO)

## Dataset

Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews):

1. Sign up on Kaggle
2. Download the IMDB dataset
3. Extract and place `IMDB Dataset.csv` in the `data/` directory

## Performance

- **Cached requests**: ~10ms response time
- **Uncached requests**: ~50-100ms response time
- **Batch predictions**: Scales linearly with batch size

## Testing

Run tests:

```bash
pytest tests/
```

## Monitoring

Access the interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

### Database connection error
- Ensure PostgreSQL is running
- Check `DATABASE_URL` environment variable
- Run migrations if needed

### Redis connection error
- Ensure Redis is running
- Check `REDIS_HOST` and `REDIS_PORT` environment variables
- The API will fallback to in-memory cache if Redis is unavailable

### Model not found error
- Train the model: `python -m app.ml.train`
- Ensure `models/sentiment_model.pkl` exists

## Future Enhancements

- [ ] Advanced NLP models (BERT, GPT)
- [ ] Real-time model updates
- [ ] A/B testing framework
- [ ] Enhanced logging and monitoring
- [ ] Kubernetes deployment
- [ ] WebSocket support for streaming predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues and questions, please open an issue on GitHub.
